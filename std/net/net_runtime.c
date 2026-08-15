#define _POSIX_C_SOURCE 200809L
#include "net_runtime.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

struct ocean_socket_handle { int fd; };

struct ocean_http_response {
    int status;
    char *status_text;
    char *headers;
    char *body;
};

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} ocean_buffer;

static void die_msg(const char *op, const char *msg) {
    fprintf(stderr, "Ocean net error: %s: %s\n", op, msg ? msg : "error");
    exit(1);
}

static void die_errno(const char *op) {
    fprintf(stderr, "Ocean net error: %s: %s\n", op, strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) die_msg("malloc", "out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) die_msg("realloc", "out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void buf_init(ocean_buffer *b) {
    b->capacity = 4096;
    b->size = 0;
    b->data = xmalloc(b->capacity);
    b->data[0] = '\0';
}

static void buf_append(ocean_buffer *b, const void *src, size_t n) {
    if (!n) return;
    size_t need = b->size + n + 1;
    if (need > b->capacity) {
        size_t cap = b->capacity;
        while (cap < need) cap *= 2;
        b->data = xrealloc(b->data, cap);
        b->capacity = cap;
    }
    memcpy(b->data + b->size, src, n);
    b->size += n;
    b->data[b->size] = '\0';
}

static void buf_cstr(ocean_buffer *b, const char *s) {
    if (s) buf_append(b, s, strlen(s));
}

static int connect_fd(const char *host, int port) {
    if (!host || !*host) die_msg("connect", "empty host");
    if (port <= 0 || port > 65535) die_msg("connect", "invalid port");

    char port_text[16];
    snprintf(port_text, sizeof(port_text), "%d", port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    struct addrinfo *list = NULL;
    int rc = getaddrinfo(host, port_text, &hints, &list);
    if (rc != 0) die_msg("getaddrinfo", gai_strerror(rc));

    int last_errno = ECONNREFUSED;
    for (struct addrinfo *it = list; it; it = it->ai_next) {
        int fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) { last_errno = errno; continue; }
        if (connect(fd, it->ai_addr, it->ai_addrlen) == 0) {
            freeaddrinfo(list);
            return fd;
        }
        last_errno = errno;
        close(fd);
    }

    freeaddrinfo(list);
    errno = last_errno;
    die_errno("connect");
    return -1;
}

static void require_socket(ocean_socket_handle_t s) {
    if (!s || s->fd < 0) die_msg("socket", "closed socket");
}

ocean_socket_handle_t ocean_socket_create(void) {
    ocean_socket_handle_t s = xmalloc(sizeof(*s));
    s->fd = -1;
    return s;
}

void ocean_socket_connect(ocean_socket_handle_t s, const char *host, int port) {
    if (!s) die_msg("connect", "null socket");
    if (s->fd >= 0) die_msg("connect", "socket already open");
    s->fd = connect_fd(host, port);
}

void ocean_socket_bind(ocean_socket_handle_t s, const char *host, int port, bool reuse_address) {
    if (!s) die_msg("bind", "null socket");
    if (s->fd >= 0) die_msg("bind", "socket already open");
    if (port < 0 || port > 65535) die_msg("bind", "invalid port");

    char port_text[16];
    snprintf(port_text, sizeof(port_text), "%d", port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *list = NULL;
    const char *bind_host = (host && *host) ? host : NULL;
    int rc = getaddrinfo(bind_host, port_text, &hints, &list);
    if (rc != 0) die_msg("getaddrinfo", gai_strerror(rc));

    int last_errno = EADDRNOTAVAIL;
    for (struct addrinfo *it = list; it; it = it->ai_next) {
        int fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) { last_errno = errno; continue; }

        if (reuse_address) {
            int yes = 1;
            (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        }

        if (bind(fd, it->ai_addr, it->ai_addrlen) == 0) {
            s->fd = fd;
            freeaddrinfo(list);
            return;
        }

        last_errno = errno;
        close(fd);
    }

    freeaddrinfo(list);
    errno = last_errno;
    die_errno("bind");
}

void ocean_socket_listen(ocean_socket_handle_t s, int backlog) {
    require_socket(s);
    if (backlog <= 0) backlog = 128;
    if (listen(s->fd, backlog) != 0) die_errno("listen");
}

ocean_socket_handle_t ocean_socket_accept(ocean_socket_handle_t s) {
    require_socket(s);
    int fd;
    do { fd = accept(s->fd, NULL, NULL); } while (fd < 0 && errno == EINTR);
    if (fd < 0) die_errno("accept");
    ocean_socket_handle_t c = ocean_socket_create();
    c->fd = fd;
    return c;
}

int ocean_socket_send(ocean_socket_handle_t s, const char *data) {
    require_socket(s);
    if (!data) return 0;

    size_t n = strlen(data);
    size_t sent_total = 0;
    while (sent_total < n) {
        ssize_t sent = send(s->fd, data + sent_total, n - sent_total, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) die_errno("send");
        sent_total += (size_t)sent;
    }
    return sent_total > 2147483647U ? 2147483647 : (int)sent_total;
}

char *ocean_socket_recv(ocean_socket_handle_t s, int max_bytes) {
    require_socket(s);
    if (max_bytes <= 0) max_bytes = 4096;

    char *buffer = xmalloc((size_t)max_bytes + 1);
    ssize_t n;
    do { n = recv(s->fd, buffer, (size_t)max_bytes, 0); } while (n < 0 && errno == EINTR);
    if (n < 0) { free(buffer); die_errno("recv"); }
    buffer[n] = '\0';
    return buffer;
}

void ocean_socket_set_timeout(ocean_socket_handle_t s, int timeout_ms) {
    require_socket(s);
    if (timeout_ms < 0) die_msg("set_timeout", "timeout must be >= 0");

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    if (setsockopt(s->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) die_errno("SO_RCVTIMEO");
    if (setsockopt(s->fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) die_errno("SO_SNDTIMEO");
}

bool ocean_socket_is_open(ocean_socket_handle_t s) {
    return s && s->fd >= 0;
}

static char *socket_address(ocean_socket_handle_t s, bool peer) {
    require_socket(s);

    struct sockaddr_storage addr;
    socklen_t len = sizeof(addr);
    int rc = peer
        ? getpeername(s->fd, (struct sockaddr *)&addr, &len)
        : getsockname(s->fd, (struct sockaddr *)&addr, &len);
    if (rc != 0) die_errno(peer ? "getpeername" : "getsockname");

    char host[128];
    char service[32];
    rc = getnameinfo(
        (struct sockaddr *)&addr, len,
        host, sizeof(host),
        service, sizeof(service),
        NI_NUMERICHOST | NI_NUMERICSERV
    );
    if (rc != 0) die_msg("getnameinfo", gai_strerror(rc));

    size_t size = strlen(host) + strlen(service) + 4;
    char *out = xmalloc(size);
    if (addr.ss_family == AF_INET6) snprintf(out, size, "[%s]:%s", host, service);
    else snprintf(out, size, "%s:%s", host, service);
    return out;
}

char *ocean_socket_peer_address(ocean_socket_handle_t s) { return socket_address(s, true); }
char *ocean_socket_local_address(ocean_socket_handle_t s) { return socket_address(s, false); }

void ocean_socket_close(ocean_socket_handle_t s) {
    if (!s || s->fd < 0) return;
    int fd = s->fd;
    s->fd = -1;
    if (close(fd) != 0 && errno != EINTR) die_errno("close");
}

void ocean_socket_release(ocean_socket_handle_t s) {
    if (!s) return;
    ocean_socket_close(s);
    free(s);
}

/* ---------------- HTTP/1.1 ---------------- */

typedef struct {
    char *host;
    int port;
    char *path;
} parsed_url;

static parsed_url parse_url(const char *url) {
    if (!url) die_msg("HTTP", "null URL");
    if (strncmp(url, "https://", 8) == 0) {
        die_msg("HTTP", "https:// is not supported in std/net v1; TLS backend required");
    }
    if (strncmp(url, "http://", 7) != 0) die_msg("HTTP", "URL must start with http://");

    const char *authority = url + 7;
    const char *slash = strchr(authority, '/');
    const char *end = slash ? slash : url + strlen(url);
    if (authority == end) die_msg("HTTP", "empty host");

    const char *colon = NULL;
    for (const char *p = authority; p < end; ++p) if (*p == ':') colon = p;

    parsed_url result;
    result.port = 80;

    if (colon) {
        result.host = xstrndup(authority, (size_t)(colon - authority));
        char *port_text = xstrndup(colon + 1, (size_t)(end - colon - 1));
        result.port = atoi(port_text);
        free(port_text);
        if (result.port <= 0 || result.port > 65535) {
            free(result.host);
            die_msg("HTTP", "invalid port");
        }
    } else {
        result.host = xstrndup(authority, (size_t)(end - authority));
    }

    result.path = slash ? xstrdup(slash) : xstrdup("/");
    return result;
}

static bool header_has(const char *headers, const char *name) {
    if (!headers || !name) return false;
    size_t name_len = strlen(name);
    const char *line = headers;

    while (*line) {
        if (strncasecmp(line, name, name_len) == 0 && line[name_len] == ':') return true;
        const char *next = strstr(line, "\r\n");
        if (!next) break;
        line = next + 2;
    }
    return false;
}

static void send_all_fd(int fd, const char *data, size_t n) {
    size_t sent_total = 0;
    while (sent_total < n) {
        ssize_t sent = send(fd, data + sent_total, n - sent_total, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) die_errno("HTTP send");
        sent_total += (size_t)sent;
    }
}

static char *recv_all_fd(int fd) {
    ocean_buffer b;
    buf_init(&b);
    char chunk[8192];

    for (;;) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { free(b.data); die_errno("HTTP recv"); }
        if (n == 0) break;
        buf_append(&b, chunk, (size_t)n);
    }
    return b.data;
}

static bool contains_ci(const char *text, size_t length, const char *needle) {
    size_t n = strlen(needle);
    if (n > length) return false;
    for (size_t i = 0; i + n <= length; ++i) {
        if (strncasecmp(text + i, needle, n) == 0) return true;
    }
    return false;
}

static bool headers_chunked(const char *headers) {
    const char *line = headers;
    while (line && *line) {
        const char *end = strstr(line, "\r\n");
        size_t len = end ? (size_t)(end - line) : strlen(line);
        const char *name = "Transfer-Encoding:";
        size_t name_len = strlen(name);

        if (len >= name_len && strncasecmp(line, name, name_len) == 0) {
            const char *value = line + name_len;
            while (value < line + len && (*value == ' ' || *value == '\t')) ++value;
            if (contains_ci(value, (size_t)(line + len - value), "chunked")) return true;
        }

        if (!end) break;
        line = end + 2;
    }
    return false;
}

static char *decode_chunked(const char *body) {
    ocean_buffer out;
    buf_init(&out);
    const char *p = body;

    for (;;) {
        const char *line_end = strstr(p, "\r\n");
        if (!line_end) { free(out.data); die_msg("HTTP", "invalid chunk header"); }

        char *size_text = xstrndup(p, (size_t)(line_end - p));
        char *semi = strchr(size_text, ';');
        if (semi) *semi = '\0';

        char *endptr = NULL;
        unsigned long size = strtoul(size_text, &endptr, 16);
        bool valid = endptr != size_text && *endptr == '\0';
        free(size_text);
        if (!valid) { free(out.data); die_msg("HTTP", "invalid chunk size"); }

        p = line_end + 2;
        if (size == 0) break;

        if (strlen(p) < size + 2) { free(out.data); die_msg("HTTP", "truncated chunk"); }
        buf_append(&out, p, (size_t)size);
        p += size;
        if (p[0] != '\r' || p[1] != '\n') { free(out.data); die_msg("HTTP", "invalid chunk ending"); }
        p += 2;
    }

    return out.data;
}

static ocean_http_response_t parse_response(char *raw) {
    char *status_end = strstr(raw, "\r\n");
    if (!status_end) { free(raw); die_msg("HTTP", "missing status line"); }

    char *headers_end = strstr(status_end + 2, "\r\n\r\n");
    if (!headers_end) { free(raw); die_msg("HTTP", "missing headers"); }

    char *status_line = xstrndup(raw, (size_t)(status_end - raw));
    char *first = strchr(status_line, ' ');
    if (!first) { free(status_line); free(raw); die_msg("HTTP", "bad status line"); }
    char *second = strchr(first + 1, ' ');

    int status = atoi(first + 1);
    char *status_text = second ? xstrdup(second + 1) : xstrdup("");
    free(status_line);

    char *headers = xstrndup(status_end + 2, (size_t)(headers_end - status_end - 2));
    char *body_start = headers_end + 4;
    char *body = headers_chunked(headers) ? decode_chunked(body_start) : xstrdup(body_start);

    ocean_http_response_t r = xmalloc(sizeof(*r));
    r->status = status;
    r->status_text = status_text;
    r->headers = headers;
    r->body = body;

    free(raw);
    return r;
}

ocean_http_response_t ocean_http_request(
    const char *method,
    const char *url,
    const char *headers,
    const char *body,
    int timeout_ms
) {
    if (!method || !*method) die_msg("HTTP", "empty method");

    parsed_url u = parse_url(url);
    int fd = connect_fd(u.host, u.port);

    if (timeout_ms > 0) {
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

    ocean_buffer req;
    buf_init(&req);
    buf_cstr(&req, method);
    buf_cstr(&req, " ");
    buf_cstr(&req, u.path);
    buf_cstr(&req, " HTTP/1.1\r\nHost: ");
    buf_cstr(&req, u.host);

    if (u.port != 80) {
        char p[24];
        snprintf(p, sizeof(p), ":%d", u.port);
        buf_cstr(&req, p);
    }

    buf_cstr(&req, "\r\n");

    if (!header_has(headers, "User-Agent")) buf_cstr(&req, "User-Agent: Ocean/0.1\r\n");
    if (!header_has(headers, "Accept")) buf_cstr(&req, "Accept: */*\r\n");
    if (!header_has(headers, "Connection")) buf_cstr(&req, "Connection: close\r\n");

    if (headers && *headers) {
        buf_cstr(&req, headers);
        size_t n = strlen(headers);
        if (n < 2 || strcmp(headers + n - 2, "\r\n") != 0) buf_cstr(&req, "\r\n");
    }

    const char *safe_body = body ? body : "";
    size_t body_len = strlen(safe_body);

    if (body_len > 0 && !header_has(headers, "Content-Length")) {
        char h[64];
        snprintf(h, sizeof(h), "Content-Length: %zu\r\n", body_len);
        buf_cstr(&req, h);
    }

    buf_cstr(&req, "\r\n");
    if (body_len) buf_append(&req, safe_body, body_len);

    send_all_fd(fd, req.data, req.size);
    free(req.data);

    char *raw = recv_all_fd(fd);
    close(fd);

    free(u.host);
    free(u.path);

    return parse_response(raw);
}

int ocean_http_status(ocean_http_response_t r) {
    if (!r) die_msg("HTTP", "null response");
    return r->status;
}

bool ocean_http_ok(ocean_http_response_t r) {
    return r && r->status >= 200 && r->status < 300;
}

char *ocean_http_status_text_copy(ocean_http_response_t r) {
    if (!r) die_msg("HTTP", "null response");
    return xstrdup(r->status_text);
}

char *ocean_http_headers_copy(ocean_http_response_t r) {
    if (!r) die_msg("HTTP", "null response");
    return xstrdup(r->headers);
}

char *ocean_http_body_copy(ocean_http_response_t r) {
    if (!r) die_msg("HTTP", "null response");
    return xstrdup(r->body);
}

void ocean_http_response_release(ocean_http_response_t r) {
    if (!r) return;
    free(r->status_text);
    free(r->headers);
    free(r->body);
    free(r);
}

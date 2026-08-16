#define _POSIX_C_SOURCE 200809L
#include "web_runtime.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define MAX_HEADER_BYTES (64 * 1024)
#define DEFAULT_MAX_BODY_BYTES (10 * 1024 * 1024)
#define DEFAULT_WORKERS 4
#define DEFAULT_QUEUE_SIZE 256
#define DEFAULT_KEEP_ALIVE_MS 5000
#define DEFAULT_MAX_KEEP_ALIVE_REQUESTS 100

typedef struct {
    char *method;
    char *pattern;
    ocean_web_handler_t handler;
} route_t;

typedef struct header_node {
    char *name;
    char *value;
    struct header_node *next;
} header_node;

typedef struct {
    int fd;
    struct sockaddr_storage remote;
    socklen_t remote_length;
} connection_t;

typedef struct {
    connection_t *items;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
    bool stopping;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} connection_queue_t;

typedef struct {
    size_t refcount;
    void (*destroy)(void *);
} ocean_arc_header;

typedef struct {
    ocean_web_app_t app;
    connection_queue_t *queue;
} worker_context_t;

struct ocean_web_app {
    route_t *routes;
    size_t route_count;
    size_t route_capacity;
    ocean_web_middleware_t *middlewares;
    size_t middleware_count;
    size_t middleware_capacity;
    char *server_header;
    int max_body_bytes;
    int workers;
    int queue_size;
    int keep_alive_timeout_ms;
    int max_keep_alive_requests;
};

struct ocean_web_request {
    char *method;
    char *path;
    char *query;
    char *body;
    char *headers;
    char *remote;
    char *version;
    const char *matched_pattern;
};

struct ocean_web_response {
    int status;
    char *content_type;
    char *body;
    header_node *headers;
};

struct ocean_web_next {
    ocean_web_app_t app;
    ocean_web_request_t request;
    ocean_Request *request_object;
    route_t *route;
    size_t next_middleware;
};

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} buffer_t;

static void die(const char *op, const char *message) {
    fprintf(stderr, "Ocean web error: %s: %s\n", op, message ? message : "error");
    exit(1);
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) die("malloc", "out of memory");
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (!next) die("realloc", "out of memory");
    return next;
}

static char *xstrdup(const char *value) {
    const char *src = value ? value : "";
    size_t n = strlen(src);
    char *copy = xmalloc(n + 1);
    memcpy(copy, src, n + 1);
    return copy;
}

static char *xstrndup(const char *value, size_t n) {
    char *copy = xmalloc(n + 1);
    memcpy(copy, value, n);
    copy[n] = '\0';
    return copy;
}

static void release_ocean_object(void *ptr) {
    if (!ptr) return;
    ocean_arc_header *header = (ocean_arc_header *)ptr;
    if (header->refcount == 0) die("ownership", "release of dead Ocean object");
    header->refcount -= 1;
    if (header->refcount == 0 && header->destroy) header->destroy(ptr);
}

static void buffer_init(buffer_t *b) {
    b->capacity = 4096;
    b->size = 0;
    b->data = xmalloc(b->capacity);
    b->data[0] = '\0';
}

static void buffer_append(buffer_t *b, const void *data, size_t n) {
    if (!n) return;
    size_t need = b->size + n + 1;
    if (need > b->capacity) {
        size_t cap = b->capacity;
        while (cap < need) cap *= 2;
        b->data = xrealloc(b->data, cap);
        b->capacity = cap;
    }
    memcpy(b->data + b->size, data, n);
    b->size += n;
    b->data[b->size] = '\0';
}

static void buffer_cstr(buffer_t *b, const char *s) {
    if (s) buffer_append(b, s, strlen(s));
}

static const char *reason_phrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 201: return "Created";
        case 202: return "Accepted";
        case 204: return "No Content";
        case 301: return "Moved Permanently";
        case 302: return "Found";
        case 303: return "See Other";
        case 307: return "Temporary Redirect";
        case 308: return "Permanent Redirect";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 422: return "Unprocessable Entity";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Status";
    }
}

static int create_listener(const char *host, int port) {
    if (port <= 0 || port > 65535) die("serve", "invalid port");
    char port_text[16];
    snprintf(port_text, sizeof(port_text), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo *addresses = NULL;
    int rc = getaddrinfo((host && *host) ? host : NULL, port_text, &hints, &addresses);
    if (rc != 0) die("getaddrinfo", gai_strerror(rc));
    int last_errno = EADDRNOTAVAIL;
    for (struct addrinfo *it = addresses; it; it = it->ai_next) {
        int fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) { last_errno = errno; continue; }
        int yes = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (bind(fd, it->ai_addr, it->ai_addrlen) != 0) { last_errno = errno; close(fd); continue; }
        if (listen(fd, 256) != 0) { last_errno = errno; close(fd); continue; }
        freeaddrinfo(addresses);
        return fd;
    }
    freeaddrinfo(addresses);
    errno = last_errno;
    die("serve", strerror(errno));
    return -1;
}

static void send_all(int fd, const char *data, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t sent = send(fd, data + off, n - off, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return;
        off += (size_t)sent;
    }
}

static bool next_segment(const char **cursor, const char **start, size_t *length) {
    const char *p = *cursor;
    while (*p == '/') ++p;
    if (!*p) { *cursor = p; return false; }
    const char *begin = p;
    while (*p && *p != '/') ++p;
    *start = begin;
    *length = (size_t)(p - begin);
    *cursor = p;
    return true;
}

static bool segment_is_param(const char *segment, size_t length) {
    return length >= 3 && segment[0] == '{' && segment[length - 1] == '}';
}

static bool path_matches(const char *pattern, const char *path) {
    if (!strcmp(pattern, "/") && !strcmp(path, "/")) return true;
    const char *a = pattern;
    const char *b = path;
    for (;;) {
        const char *as, *bs;
        size_t an, bn;
        bool ah = next_segment(&a, &as, &an);
        bool bh = next_segment(&b, &bs, &bn);
        if (!ah || !bh) return ah == bh;
        if (segment_is_param(as, an)) continue;
        if (an != bn || strncmp(as, bs, an) != 0) return false;
    }
}

static char *url_decode(const char *value, size_t length) {
    char *out = xmalloc(length + 1);
    size_t w = 0;
    for (size_t i = 0; i < length; ++i) {
        if (value[i] == '%' && i + 2 < length && isxdigit((unsigned char)value[i + 1]) && isxdigit((unsigned char)value[i + 2])) {
            char hex[3] = {value[i + 1], value[i + 2], '\0'};
            out[w++] = (char)strtol(hex, NULL, 16);
            i += 2;
        } else if (value[i] == '+') {
            out[w++] = ' ';
        } else {
            out[w++] = value[i];
        }
    }
    out[w] = '\0';
    return out;
}

static char *header_value_copy(const char *headers, const char *name, const char *default_value) {
    if (!headers || !name) return xstrdup(default_value);
    size_t nl = strlen(name);
    const char *cursor = headers;
    while (*cursor) {
        const char *end = strstr(cursor, "\r\n");
        if (!end) end = cursor + strlen(cursor);
        const char *colon = memchr(cursor, ':', (size_t)(end - cursor));
        if (colon && (size_t)(colon - cursor) == nl && strncasecmp(cursor, name, nl) == 0) {
            const char *value = colon + 1;
            while (value < end && (*value == ' ' || *value == '\t')) ++value;
            return xstrndup(value, (size_t)(end - value));
        }
        if (!*end) break;
        cursor = end + 2;
    }
    return xstrdup(default_value);
}

static long content_length(const char *headers) {
    char *value = header_value_copy(headers, "Content-Length", "0");
    char *end = NULL;
    long result = strtol(value, &end, 10);
    bool ok = end != value && *end == '\0' && result >= 0;
    free(value);
    return ok ? result : -1;
}

static char *pair_value(const char *pairs, const char *name, const char *default_value) {
    if (!pairs || !name) return xstrdup(default_value);
    size_t nl = strlen(name);
    const char *cursor = pairs;
    while (*cursor) {
        const char *end = strchr(cursor, '&');
        if (!end) end = cursor + strlen(cursor);
        const char *eq = memchr(cursor, '=', (size_t)(end - cursor));
        const char *key_end = eq ? eq : end;
        if ((size_t)(key_end - cursor) == nl && strncmp(cursor, name, nl) == 0) {
            return eq ? url_decode(eq + 1, (size_t)(end - eq - 1)) : xstrdup("");
        }
        if (!*end) break;
        cursor = end + 1;
    }
    return xstrdup(default_value);
}

static char *remote_copy(const struct sockaddr_storage *address, socklen_t length) {
    char host[128], service[32];
    if (getnameinfo((const struct sockaddr *)address, length, host, sizeof(host), service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV) != 0) return xstrdup("");
    size_t n = strlen(host) + strlen(service) + 4;
    char *out = xmalloc(n);
    snprintf(out, n, address->ss_family == AF_INET6 ? "[%s]:%s" : "%s:%s", host, service);
    return out;
}

static void set_socket_timeout(int fd, int timeout_ms) {
    if (timeout_ms <= 0) return;
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static ocean_web_request_t read_request(int fd, const struct sockaddr_storage *remote, socklen_t remote_length, int max_body_bytes, int *error_status) {
    *error_status = 0;
    buffer_t buffer;
    buffer_init(&buffer);
    char chunk[4096];
    char *headers_end = NULL;
    while (!headers_end) {
        ssize_t received = recv(fd, chunk, sizeof(chunk), 0);
        if (received < 0 && errno == EINTR) continue;
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { free(buffer.data); *error_status = 408; return NULL; }
        if (received == 0) { free(buffer.data); return NULL; }
        if (received < 0) { free(buffer.data); *error_status = 400; return NULL; }
        buffer_append(&buffer, chunk, (size_t)received);
        if (buffer.size > MAX_HEADER_BYTES) { free(buffer.data); *error_status = 413; return NULL; }
        headers_end = strstr(buffer.data, "\r\n\r\n");
    }

    size_t header_bytes = (size_t)(headers_end - buffer.data) + 4;
    char *line_end = strstr(buffer.data, "\r\n");
    if (!line_end) { free(buffer.data); *error_status = 400; return NULL; }
    char *line = xstrndup(buffer.data, (size_t)(line_end - buffer.data));
    char *s1 = strchr(line, ' ');
    char *s2 = s1 ? strchr(s1 + 1, ' ') : NULL;
    if (!s1 || !s2) { free(line); free(buffer.data); *error_status = 400; return NULL; }
    *s1 = '\0';
    *s2 = '\0';
    char *method = xstrdup(line);
    char *target = xstrdup(s1 + 1);
    char *version = xstrdup(s2 + 1);
    free(line);

    char *headers = xstrndup(line_end + 2, (size_t)(headers_end - (line_end + 2)));
    long body_length = content_length(headers);
    if (body_length < 0 || body_length > max_body_bytes) {
        free(method); free(target); free(version); free(headers); free(buffer.data);
        *error_status = body_length > max_body_bytes ? 413 : 400;
        return NULL;
    }

    size_t have = buffer.size - header_bytes;
    while (have < (size_t)body_length) {
        ssize_t received = recv(fd, chunk, sizeof(chunk), 0);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) {
            free(method); free(target); free(version); free(headers); free(buffer.data);
            *error_status = (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) ? 408 : 400;
            return NULL;
        }
        buffer_append(&buffer, chunk, (size_t)received);
        have += (size_t)received;
    }

    char *body = xstrndup(buffer.data + header_bytes, (size_t)body_length);
    free(buffer.data);
    char *qmark = strchr(target, '?');
    char *path = qmark ? xstrndup(target, (size_t)(qmark - target)) : xstrdup(target);
    char *query = qmark ? xstrdup(qmark + 1) : xstrdup("");
    free(target);

    ocean_web_request_t request = xmalloc(sizeof(*request));
    request->method = method;
    request->path = path;
    request->query = query;
    request->body = body;
    request->headers = headers;
    request->remote = remote_copy(remote, remote_length);
    request->version = version;
    request->matched_pattern = NULL;
    return request;
}

static void request_release(ocean_web_request_t request) {
    if (!request) return;
    free(request->method); free(request->path); free(request->query); free(request->body);
    free(request->headers); free(request->remote); free(request->version); free(request);
}

char *ocean_web_request_method_copy(ocean_web_request_t r) { return xstrdup(r ? r->method : ""); }
char *ocean_web_request_path_copy(ocean_web_request_t r) { return xstrdup(r ? r->path : ""); }
char *ocean_web_request_query_copy(ocean_web_request_t r) { return xstrdup(r ? r->query : ""); }
char *ocean_web_request_body_copy(ocean_web_request_t r) { return xstrdup(r ? r->body : ""); }
char *ocean_web_request_remote_copy(ocean_web_request_t r) { return xstrdup(r ? r->remote : ""); }
char *ocean_web_request_header_copy(ocean_web_request_t r, const char *n, const char *d) { return r ? header_value_copy(r->headers, n, d) : xstrdup(d); }
char *ocean_web_request_query_param_copy(ocean_web_request_t r, const char *n, const char *d) { return r ? pair_value(r->query, n, d) : xstrdup(d); }

char *ocean_web_request_path_param_copy(ocean_web_request_t r, const char *name, const char *default_value) {
    if (!r || !r->matched_pattern) return xstrdup(default_value);
    const char *a = r->matched_pattern;
    const char *b = r->path;
    size_t nl = strlen(name);
    for (;;) {
        const char *as, *bs;
        size_t an, bn;
        bool ah = next_segment(&a, &as, &an);
        bool bh = next_segment(&b, &bs, &bn);
        if (!ah || !bh) break;
        if (segment_is_param(as, an) && an - 2 == nl && strncmp(as + 1, name, nl) == 0) return url_decode(bs, bn);
    }
    return xstrdup(default_value);
}

static ocean_web_response_t make_response(int status, const char *content_type, const char *body) {
    ocean_web_response_t r = xmalloc(sizeof(*r));
    r->status = status;
    r->content_type = xstrdup(content_type);
    r->body = xstrdup(body);
    r->headers = NULL;
    return r;
}

ocean_web_response_t ocean_web_response_text(int s, const char *b) { return make_response(s, "text/plain; charset=utf-8", b); }
ocean_web_response_t ocean_web_response_json(int s, const char *b) { return make_response(s, "application/json; charset=utf-8", b); }
ocean_web_response_t ocean_web_response_html(int s, const char *b) { return make_response(s, "text/html; charset=utf-8", b); }
ocean_web_response_t ocean_web_response_empty(int s) { return make_response(s, "", ""); }
ocean_web_response_t ocean_web_response_redirect(int s, const char *location) {
    ocean_web_response_t r = make_response(s, "text/plain; charset=utf-8", "");
    ocean_web_response_add_header(r, "Location", location);
    return r;
}

void ocean_web_response_add_header(ocean_web_response_t r, const char *name, const char *value) {
    if (!r || !name) return;
    header_node *h = xmalloc(sizeof(*h));
    h->name = xstrdup(name); h->value = xstrdup(value); h->next = r->headers; r->headers = h;
}

void ocean_web_response_release(ocean_web_response_t r) {
    if (!r) return;
    header_node *h = r->headers;
    while (h) { header_node *next = h->next; free(h->name); free(h->value); free(h); h = next; }
    free(r->content_type); free(r->body); free(r);
}

static bool has_response_header(ocean_web_response_t r, const char *name) {
    for (header_node *h = r->headers; h; h = h->next) if (!strcasecmp(h->name, name)) return true;
    return false;
}

static bool request_keep_alive(ocean_web_request_t request) {
    char *connection = header_value_copy(request->headers, "Connection", "");
    bool close_requested = !strcasecmp(connection, "close");
    bool keep_requested = !strcasecmp(connection, "keep-alive");
    bool http11 = !strcmp(request->version, "HTTP/1.1");
    free(connection);
    if (close_requested) return false;
    return http11 || keep_requested;
}

static void write_response(int fd, ocean_web_app_t app, ocean_web_response_t r, bool head, bool keep_alive, int remaining) {
    bool owned = false;
    if (!r) { r = ocean_web_response_text(500, "handler returned null response"); owned = true; keep_alive = false; }
    buffer_t out;
    buffer_init(&out);
    char line[128];
    snprintf(line, sizeof(line), "HTTP/1.1 %d %s\r\n", r->status, reason_phrase(r->status));
    buffer_cstr(&out, line);
    if (app->server_header && *app->server_header && !has_response_header(r, "Server")) {
        buffer_cstr(&out, "Server: "); buffer_cstr(&out, app->server_header); buffer_cstr(&out, "\r\n");
    }
    if (r->content_type && *r->content_type && !has_response_header(r, "Content-Type")) {
        buffer_cstr(&out, "Content-Type: "); buffer_cstr(&out, r->content_type); buffer_cstr(&out, "\r\n");
    }
    for (header_node *h = r->headers; h; h = h->next) {
        if (!strcasecmp(h->name, "Connection") || !strcasecmp(h->name, "Keep-Alive") || !strcasecmp(h->name, "Content-Length")) continue;
        buffer_cstr(&out, h->name); buffer_cstr(&out, ": "); buffer_cstr(&out, h->value); buffer_cstr(&out, "\r\n");
    }
    size_t body_len = strlen(r->body ? r->body : "");
    {
        char tmp[64]; snprintf(tmp, sizeof(tmp), "Content-Length: %zu\r\n", body_len); buffer_cstr(&out, tmp);
    }
    buffer_cstr(&out, keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n");
    if (keep_alive) {
        char tmp[96];
        snprintf(tmp, sizeof(tmp), "Keep-Alive: timeout=%d, max=%d\r\n", app->keep_alive_timeout_ms / 1000, remaining);
        buffer_cstr(&out, tmp);
    }
    buffer_cstr(&out, "\r\n");
    if (!head) buffer_append(&out, r->body, body_len);
    send_all(fd, out.data, out.size);
    free(out.data);
    if (owned) ocean_web_response_release(r);
}

static void reserve_routes(ocean_web_app_t app) {
    if (app->route_count < app->route_capacity) return;
    size_t cap = app->route_capacity ? app->route_capacity * 2 : 16;
    app->routes = xrealloc(app->routes, cap * sizeof(*app->routes));
    app->route_capacity = cap;
}

static void reserve_middlewares(ocean_web_app_t app) {
    if (app->middleware_count < app->middleware_capacity) return;
    size_t cap = app->middleware_capacity ? app->middleware_capacity * 2 : 8;
    app->middlewares = xrealloc(app->middlewares, cap * sizeof(*app->middlewares));
    app->middleware_capacity = cap;
}

ocean_web_app_t ocean_web_app_create(void) {
    ocean_web_app_t app = xmalloc(sizeof(*app));
    memset(app, 0, sizeof(*app));
    app->server_header = xstrdup("Ocean");
    app->max_body_bytes = DEFAULT_MAX_BODY_BYTES;
    app->workers = DEFAULT_WORKERS;
    app->queue_size = DEFAULT_QUEUE_SIZE;
    app->keep_alive_timeout_ms = DEFAULT_KEEP_ALIVE_MS;
    app->max_keep_alive_requests = DEFAULT_MAX_KEEP_ALIVE_REQUESTS;
    return app;
}

void ocean_web_app_release(ocean_web_app_t app) {
    if (!app) return;
    for (size_t i = 0; i < app->route_count; ++i) { free(app->routes[i].method); free(app->routes[i].pattern); }
    free(app->routes); free(app->middlewares); free(app->server_header); free(app);
}

void ocean_web_route(ocean_web_app_t app, const char *method, const char *path, ocean_web_handler_t handler) {
    if (!app || !method || !path || path[0] != '/' || !handler) die("route", "invalid route");
    reserve_routes(app);
    route_t *r = &app->routes[app->route_count++];
    r->method = xstrdup(method); r->pattern = xstrdup(path); r->handler = handler;
}

#define ROUTE(fn, method_text) void fn(ocean_web_app_t app, const char *path, ocean_web_handler_t handler) { ocean_web_route(app, method_text, path, handler); }
ROUTE(ocean_web_get, "GET")
ROUTE(ocean_web_post, "POST")
ROUTE(ocean_web_put, "PUT")
ROUTE(ocean_web_patch, "PATCH")
ROUTE(ocean_web_delete, "DELETE")
ROUTE(ocean_web_options, "OPTIONS")
ROUTE(ocean_web_head, "HEAD")
ROUTE(ocean_web_any, "*")
#undef ROUTE

void ocean_web_middleware(ocean_web_app_t app, ocean_web_middleware_t middleware) {
    if (!app || !middleware) die("middleware", "invalid middleware");
    reserve_middlewares(app);
    app->middlewares[app->middleware_count++] = middleware;
}

void ocean_web_set_server_header(ocean_web_app_t app, const char *value) { if (app) { free(app->server_header); app->server_header = xstrdup(value); } }
void ocean_web_set_max_body_bytes(ocean_web_app_t app, int value) { if (!app || value <= 0) die("max_body", "value must be > 0"); app->max_body_bytes = value; }
void ocean_web_set_workers(ocean_web_app_t app, int value) { if (!app || value <= 0 || value > 1024) die("workers", "value must be 1..1024"); app->workers = value; }
void ocean_web_set_queue_size(ocean_web_app_t app, int value) { if (!app || value <= 0) die("queue_size", "value must be > 0"); app->queue_size = value; }
void ocean_web_set_keep_alive_timeout(ocean_web_app_t app, int value) { if (!app || value < 0) die("keep_alive", "timeout must be >= 0"); app->keep_alive_timeout_ms = value; }
void ocean_web_set_max_keep_alive_requests(ocean_web_app_t app, int value) { if (!app || value <= 0) die("max_keep_alive_requests", "value must be > 0"); app->max_keep_alive_requests = value; }

static route_t *find_route(ocean_web_app_t app, ocean_web_request_t req, bool *path_exists) {
    *path_exists = false;
    route_t *head = NULL;
    for (size_t i = 0; i < app->route_count; ++i) {
        route_t *r = &app->routes[i];
        if (!path_matches(r->pattern, req->path)) continue;
        *path_exists = true;
        if (!strcmp(r->method, req->method) || !strcmp(r->method, "*")) return r;
        if (!strcmp(req->method, "HEAD") && !strcmp(r->method, "GET")) head = r;
    }
    return head;
}

static ocean_Response *dispatch_chain(ocean_web_app_t app, ocean_web_request_t req, ocean_Request *request_object, route_t *route, size_t index) {
    if (index >= app->middleware_count) return route->handler(request_object);
    struct ocean_web_next ctx;
    ctx.app = app;
    ctx.request = req;
    ctx.request_object = request_object;
    ctx.route = route;
    ctx.next_middleware = index + 1;
    ocean_Next *next_object = ocean_create_Next(&ctx);
    ocean_Response *response = app->middlewares[index](request_object, next_object);
    release_ocean_object(next_object);
    return response;
}

ocean_web_response_t ocean_web_next_call(ocean_web_next_t next, ocean_web_request_t request) {
    if (!next || !request || request != next->request) return ocean_web_response_text(500, "invalid middleware continuation");
    ocean_Response *response_object = dispatch_chain(next->app, next->request, next->request_object, next->route, next->next_middleware);
    if (!response_object) return ocean_web_response_text(500, "middleware chain returned null Response");
    ocean_web_response_t response = ocean_Response_take_handle(response_object);
    release_ocean_object(response_object);
    return response ? response : ocean_web_response_text(500, "middleware returned empty Response");
}

static void serve_error(int fd, ocean_web_app_t app, int status, bool keep_alive, int remaining) {
    ocean_web_response_t r = ocean_web_response_text(status, reason_phrase(status));
    write_response(fd, app, r, false, keep_alive, remaining);
    ocean_web_response_release(r);
}

static void handle_connection(ocean_web_app_t app, connection_t *connection) {
    int fd = connection->fd;
    set_socket_timeout(fd, app->keep_alive_timeout_ms);
    for (int n = 0; n < app->max_keep_alive_requests; ++n) {
        int error_status = 0;
        ocean_web_request_t req = read_request(fd, &connection->remote, connection->remote_length, app->max_body_bytes, &error_status);
        if (!req) {
            if (error_status && error_status != 408) serve_error(fd, app, error_status, false, 0);
            break;
        }
        int remaining = app->max_keep_alive_requests - n - 1;
        bool keep_alive = app->keep_alive_timeout_ms > 0 && request_keep_alive(req) && remaining > 0;
        bool path_exists = false;
        route_t *route = find_route(app, req, &path_exists);
        if (!route) {
            serve_error(fd, app, path_exists ? 405 : 404, keep_alive, remaining);
            request_release(req);
            if (!keep_alive) break;
            continue;
        }
        req->matched_pattern = route->pattern;
        ocean_Request *request_object = ocean_create_Request(req);
        ocean_Response *response_object = dispatch_chain(app, req, request_object, route, 0);
        ocean_web_response_t response = NULL;
        if (response_object) {
            response = ocean_Response_take_handle(response_object);
            release_ocean_object(response_object);
        }
        write_response(fd, app, response, !strcmp(req->method, "HEAD"), keep_alive, remaining);
        if (response) ocean_web_response_release(response);
        release_ocean_object(request_object);
        request_release(req);
        if (!keep_alive) break;
    }
    close(fd);
}

static void queue_init(connection_queue_t *q, size_t capacity) {
    memset(q, 0, sizeof(*q));
    q->items = xmalloc(capacity * sizeof(*q->items));
    q->capacity = capacity;
    if (pthread_mutex_init(&q->mutex, NULL) != 0) die("pthread_mutex_init", "failed");
    if (pthread_cond_init(&q->not_empty, NULL) != 0) die("pthread_cond_init", "failed");
    if (pthread_cond_init(&q->not_full, NULL) != 0) die("pthread_cond_init", "failed");
}

static void queue_push(connection_queue_t *q, const connection_t *c) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == q->capacity && !q->stopping) pthread_cond_wait(&q->not_full, &q->mutex);
    if (!q->stopping) {
        q->items[q->tail] = *c;
        q->tail = (q->tail + 1) % q->capacity;
        q->count += 1;
        pthread_cond_signal(&q->not_empty);
    }
    pthread_mutex_unlock(&q->mutex);
}

static bool queue_pop(connection_queue_t *q, connection_t *c) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0 && !q->stopping) pthread_cond_wait(&q->not_empty, &q->mutex);
    if (q->count == 0 && q->stopping) { pthread_mutex_unlock(&q->mutex); return false; }
    *c = q->items[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count -= 1;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return true;
}

static void *worker_main(void *arg) {
    worker_context_t *ctx = (worker_context_t *)arg;
    connection_t connection;
    while (queue_pop(ctx->queue, &connection)) handle_connection(ctx->app, &connection);
    return NULL;
}

void ocean_web_serve(ocean_web_app_t app, const char *host, int port) {
    if (!app) die("serve", "null app");
    int server_fd = create_listener(host, port);
    connection_queue_t queue;
    queue_init(&queue, (size_t)app->queue_size);
    pthread_t *threads = xmalloc((size_t)app->workers * sizeof(*threads));
    worker_context_t context = {app, &queue};
    for (int i = 0; i < app->workers; ++i) {
        int rc = pthread_create(&threads[i], NULL, worker_main, &context);
        if (rc != 0) die("pthread_create", strerror(rc));
    }
    printf("Ocean web server listening on http://%s:%d (workers=%d, keep-alive=%dms)\n", (host && *host) ? host : "0.0.0.0", port, app->workers, app->keep_alive_timeout_ms);
    fflush(stdout);
    for (;;) {
        connection_t c;
        c.remote_length = sizeof(c.remote);
        do {
            c.fd = accept(server_fd, (struct sockaddr *)&c.remote, &c.remote_length);
        } while (c.fd < 0 && errno == EINTR);
        if (c.fd < 0) continue;
        queue_push(&queue, &c);
    }
}


/* Ocean Router: private-layout-independent implementation. */

typedef struct ocean_router_route {
    char *method;
    char *path;
    ocean_web_handler_t handler;
} ocean_router_route;

struct ocean_web_router {
    char *prefix;
    ocean_router_route *routes;
    size_t count;
    size_t capacity;
};

static void *ocean_router_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Ocean Router: out of memory\n");
        exit(1);
    }
    return ptr;
}

static void *ocean_router_realloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (!next) {
        fprintf(stderr, "Ocean Router: out of memory\n");
        exit(1);
    }
    return next;
}

static char *ocean_router_strndup(const char *src, size_t length) {
    char *copy = ocean_router_malloc(length + 1);
    memcpy(copy, src, length);
    copy[length] = '\0';
    return copy;
}

static char *ocean_router_strdup(const char *src) {
    if (!src) {
        src = "";
    }
    return ocean_router_strndup(src, strlen(src));
}

static void ocean_router_fail(const char *message) {
    fprintf(stderr, "Ocean Router error: %s\n", message);
    exit(1);
}

static char *ocean_router_normalize_prefix(const char *prefix) {
    if (!prefix || prefix[0] == '\0' || strcmp(prefix, "/") == 0) {
        return ocean_router_strdup("");
    }

    if (prefix[0] != '/') {
        ocean_router_fail("prefix must start with '/'");
    }

    size_t length = strlen(prefix);

    while (length > 1 && prefix[length - 1] == '/') {
        --length;
    }

    return ocean_router_strndup(prefix, length);
}

static char *ocean_router_join_path(
    const char *prefix,
    const char *path
) {
    if (!path || path[0] == '\0') {
        path = "/";
    }

    if (path[0] != '/') {
        ocean_router_fail("route path must start with '/'");
    }

    if (!prefix || prefix[0] == '\0') {
        return ocean_router_strdup(path);
    }

    if (strcmp(path, "/") == 0) {
        return ocean_router_strdup(prefix);
    }

    size_t prefix_length = strlen(prefix);
    size_t path_length = strlen(path);

    char *result = ocean_router_malloc(
        prefix_length + path_length + 1
    );

    memcpy(result, prefix, prefix_length);
    memcpy(result + prefix_length, path, path_length + 1);

    return result;
}

static void ocean_router_reserve(ocean_web_router_t router) {
    if (router->count < router->capacity) {
        return;
    }

    size_t next_capacity = router->capacity
        ? router->capacity * 2
        : 8;

    router->routes = ocean_router_realloc(
        router->routes,
        next_capacity * sizeof(ocean_router_route)
    );

    router->capacity = next_capacity;
}

ocean_web_router_t ocean_web_router_create(const char *prefix) {
    ocean_web_router_t router = ocean_router_malloc(
        sizeof(struct ocean_web_router)
    );

    router->prefix = ocean_router_normalize_prefix(prefix);
    router->routes = NULL;
    router->count = 0;
    router->capacity = 0;

    return router;
}

void ocean_web_router_release(ocean_web_router_t router) {
    if (!router) {
        return;
    }

    for (size_t i = 0; i < router->count; ++i) {
        free(router->routes[i].method);
        free(router->routes[i].path);
    }

    free(router->routes);
    free(router->prefix);
    free(router);
}

void ocean_web_router_route(
    ocean_web_router_t router,
    const char *method,
    const char *path,
    ocean_web_handler_t handler
) {
    if (!router || !method || !path || !handler) {
        ocean_router_fail("Router.route() received an invalid argument");
    }

    if (path[0] != '/') {
        ocean_router_fail("Router route path must start with '/'");
    }

    ocean_router_reserve(router);

    ocean_router_route *entry =
        &router->routes[router->count++];

    entry->method = ocean_router_strdup(method);
    entry->path = ocean_router_strdup(path);
    entry->handler = handler;
}

#define OCEAN_ROUTER_METHOD(function_name, method_name) \
    void function_name( \
        ocean_web_router_t router, \
        const char *path, \
        ocean_web_handler_t handler \
    ) { \
        ocean_web_router_route(router, method_name, path, handler); \
    }

OCEAN_ROUTER_METHOD(ocean_web_router_get, "GET")
OCEAN_ROUTER_METHOD(ocean_web_router_post, "POST")
OCEAN_ROUTER_METHOD(ocean_web_router_put, "PUT")
OCEAN_ROUTER_METHOD(ocean_web_router_patch, "PATCH")
OCEAN_ROUTER_METHOD(ocean_web_router_delete, "DELETE")
OCEAN_ROUTER_METHOD(ocean_web_router_options, "OPTIONS")
OCEAN_ROUTER_METHOD(ocean_web_router_head, "HEAD")
OCEAN_ROUTER_METHOD(ocean_web_router_any, "*")

#undef OCEAN_ROUTER_METHOD

void ocean_web_include_router(
    ocean_web_app_t app,
    ocean_web_router_t router
) {
    if (!app || !router) {
        ocean_router_fail("App.include() requires a valid Router");
    }

    for (size_t i = 0; i < router->count; ++i) {
        ocean_router_route *entry =
            &router->routes[i];

        char *full_path = ocean_router_join_path(
            router->prefix,
            entry->path
        );

        ocean_web_route(
            app,
            entry->method,
            full_path,
            entry->handler
        );

        free(full_path);
    }
}


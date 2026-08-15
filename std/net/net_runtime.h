#ifndef OCEAN_STD_NET_NET_RUNTIME_H
#define OCEAN_STD_NET_NET_RUNTIME_H

#include <stdbool.h>

typedef struct ocean_socket_handle *ocean_socket_handle_t;
typedef struct ocean_http_response *ocean_http_response_t;

ocean_socket_handle_t ocean_socket_create(void);
void ocean_socket_connect(ocean_socket_handle_t s, const char *host, int port);
void ocean_socket_bind(ocean_socket_handle_t s, const char *host, int port, bool reuse_address);
void ocean_socket_listen(ocean_socket_handle_t s, int backlog);
ocean_socket_handle_t ocean_socket_accept(ocean_socket_handle_t s);
int ocean_socket_send(ocean_socket_handle_t s, const char *data);
char *ocean_socket_recv(ocean_socket_handle_t s, int max_bytes);
void ocean_socket_set_timeout(ocean_socket_handle_t s, int timeout_ms);
bool ocean_socket_is_open(ocean_socket_handle_t s);
char *ocean_socket_peer_address(ocean_socket_handle_t s);
char *ocean_socket_local_address(ocean_socket_handle_t s);
void ocean_socket_close(ocean_socket_handle_t s);
void ocean_socket_release(ocean_socket_handle_t s);

ocean_http_response_t ocean_http_request(
    const char *method,
    const char *url,
    const char *headers,
    const char *body,
    int timeout_ms
);
int ocean_http_status(ocean_http_response_t r);
bool ocean_http_ok(ocean_http_response_t r);
char *ocean_http_status_text_copy(ocean_http_response_t r);
char *ocean_http_headers_copy(ocean_http_response_t r);
char *ocean_http_body_copy(ocean_http_response_t r);
void ocean_http_response_release(ocean_http_response_t r);

#endif

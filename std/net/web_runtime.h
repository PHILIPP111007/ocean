#ifndef OCEAN_STD_NET_WEB_RUNTIME_H
#define OCEAN_STD_NET_WEB_RUNTIME_H

#include <stdbool.h>

typedef struct ocean_web_app *ocean_web_app_t;
typedef struct ocean_web_request *ocean_web_request_t;
typedef struct ocean_web_response *ocean_web_response_t;
typedef struct ocean_web_next *ocean_web_next_t;

typedef struct ocean_Request ocean_Request;
typedef struct ocean_Response ocean_Response;
typedef struct ocean_Next ocean_Next;

typedef ocean_Response *(*ocean_web_handler_t)(ocean_Request *request);
typedef ocean_Response *(*ocean_web_middleware_t)(ocean_Request *request, ocean_Next *call_next);

ocean_Request *ocean_create_Request(ocean_web_request_t handle);
ocean_Next *ocean_create_Next(ocean_web_next_t handle);
ocean_web_response_t ocean_Response_take_handle(ocean_Response *response);

ocean_web_app_t ocean_web_app_create(void);
void ocean_web_app_release(ocean_web_app_t app);

void ocean_web_route(ocean_web_app_t app, const char *method, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_get(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_post(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_put(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_patch(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_delete(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_options(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_head(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);
void ocean_web_any(ocean_web_app_t app, const char *path_pattern, ocean_web_handler_t handler);

void ocean_web_middleware(ocean_web_app_t app, ocean_web_middleware_t middleware);
void ocean_web_set_server_header(ocean_web_app_t app, const char *value);
void ocean_web_set_max_body_bytes(ocean_web_app_t app, int max_body_bytes);
void ocean_web_set_workers(ocean_web_app_t app, int workers);
void ocean_web_set_queue_size(ocean_web_app_t app, int queue_size);
void ocean_web_set_keep_alive_timeout(ocean_web_app_t app, int timeout_ms);
void ocean_web_set_max_keep_alive_requests(ocean_web_app_t app, int max_requests);
void ocean_web_serve(ocean_web_app_t app, const char *host, int port);

ocean_web_response_t ocean_web_next_call(ocean_web_next_t next, ocean_web_request_t request);

char *ocean_web_request_method_copy(ocean_web_request_t request);
char *ocean_web_request_path_copy(ocean_web_request_t request);
char *ocean_web_request_query_copy(ocean_web_request_t request);
char *ocean_web_request_body_copy(ocean_web_request_t request);
char *ocean_web_request_remote_copy(ocean_web_request_t request);
char *ocean_web_request_header_copy(ocean_web_request_t request, const char *name, const char *default_value);
char *ocean_web_request_query_param_copy(ocean_web_request_t request, const char *name, const char *default_value);
char *ocean_web_request_path_param_copy(ocean_web_request_t request, const char *name, const char *default_value);

ocean_web_response_t ocean_web_response_text(int status, const char *body);
ocean_web_response_t ocean_web_response_json(int status, const char *body);
ocean_web_response_t ocean_web_response_html(int status, const char *body);
ocean_web_response_t ocean_web_response_empty(int status);
ocean_web_response_t ocean_web_response_redirect(int status, const char *location);
void ocean_web_response_add_header(ocean_web_response_t response, const char *name, const char *value);
void ocean_web_response_release(ocean_web_response_t response);

#endif

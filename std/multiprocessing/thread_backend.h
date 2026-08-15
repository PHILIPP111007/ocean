#ifndef OCEAN_STD_THREAD_BACKEND_H
#define OCEAN_STD_THREAD_BACKEND_H

#include <stdbool.h>
#include <pthread.h>

typedef struct ocean_thread_handle *ocean_thread_handle_t;
typedef void *(*ocean_thread_fn_t)(void *);

ocean_thread_handle_t ocean_thread_create(ocean_thread_fn_t start_routine, void *arg);
void ocean_thread_join(ocean_thread_handle_t thread);
void ocean_thread_detach(ocean_thread_handle_t thread);
bool ocean_thread_is_joinable(ocean_thread_handle_t thread);
void ocean_thread_release(ocean_thread_handle_t thread);

#endif

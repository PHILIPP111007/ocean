#include "thread_backend.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct ocean_thread_handle { pthread_t id; bool joinable; };
static void ocean_thread_fail(const char *operation, int error_code) {
    fprintf(stderr, "Ocean thread error: %s failed: %s (%d)\n", operation, strerror(error_code), error_code);
    abort();
}
ocean_thread_handle_t ocean_thread_create(ocean_thread_fn_t start_routine, void *arg) {
    if (!start_routine) ocean_thread_fail("pthread_create", EINVAL);
    ocean_thread_handle_t thread = calloc(1, sizeof(*thread));
    if (!thread) { fprintf(stderr, "Ocean thread error: out of memory\n"); abort(); }
    const int result = pthread_create(&thread->id, NULL, start_routine, arg);
    if (result != 0) { free(thread); ocean_thread_fail("pthread_create", result); }
    thread->joinable = true;
    return thread;
}
void ocean_thread_join(ocean_thread_handle_t thread) {
    if (!thread || !thread->joinable) ocean_thread_fail("pthread_join", EINVAL);
    const int result = pthread_join(thread->id, NULL);
    if (result != 0) ocean_thread_fail("pthread_join", result);
    thread->joinable = false;
}
void ocean_thread_detach(ocean_thread_handle_t thread) {
    if (!thread || !thread->joinable) ocean_thread_fail("pthread_detach", EINVAL);
    const int result = pthread_detach(thread->id);
    if (result != 0) ocean_thread_fail("pthread_detach", result);
    thread->joinable = false;
}
bool ocean_thread_is_joinable(ocean_thread_handle_t thread) { return thread && thread->joinable; }
void ocean_thread_release(ocean_thread_handle_t thread) {
    if (!thread) return;
    if (thread->joinable) {
        const int result = pthread_detach(thread->id);
        if (result != 0 && result != EINVAL && result != ESRCH) {
            fprintf(stderr, "Ocean thread warning: pthread_detach: %s (%d)\n", strerror(result), result);
        }
        thread->joinable = false;
    }
    free(thread);
}

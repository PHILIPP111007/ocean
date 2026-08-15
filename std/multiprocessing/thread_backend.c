#include "thread_backend.h"

int ocean_thread_create(ocean_Thread* thread, void* (*start_routine)(void*), void* arg) {
    return pthread_create(&thread->id, NULL, start_routine, arg);
}

int ocean_thread_join(ocean_Thread* thread) {
    return pthread_join(thread->id, NULL);
}
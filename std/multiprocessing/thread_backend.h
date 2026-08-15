#ifndef OCEAN_STD_THREAD_BACKEND_H
#define OCEAN_STD_THREAD_BACKEND_H

#include <pthread.h>

// Структура, содержащая pthread_t (скрываем реализацию)
typedef struct ocean_Thread {
    pthread_t id;
} ocean_Thread;

// Создаёт поток и запускает функцию с аргументом
int ocean_thread_create(ocean_Thread* thread, void* (*start_routine)(void*), void* arg);

// Ожидает завершения потока
int ocean_thread_join(ocean_Thread* thread);

#endif
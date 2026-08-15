#ifndef OCEAN_STD_THREAD_BACKEND_H
#define OCEAN_STD_THREAD_BACKEND_H

#include <pthread.h>

// Структура для хранения данных потока
typedef struct {
    void* (*start_routine)(void*);  // указатель на функцию потока
    void* arg;                      // аргумент для функции
} ocean_thread_task_t;

// Создаёт поток и запускает переданную функцию с аргументом
int ocean_thread_create(pthread_t* thread, void* (*start_routine)(void*), void* arg);

// Ожидает завершения потока
int ocean_thread_join(pthread_t thread);

// Функция-обёртка, которая извлекает задачу из структуры и запускает её
void* ocean_thread_dispatch(void* task_ptr);

#endif // OCEAN_STD_THREAD_BACKEND_H

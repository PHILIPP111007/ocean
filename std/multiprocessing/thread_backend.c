#include "std/multiprocessing/thread_backend.h"


// Создание потока
int ocean_thread_create(pthread_t* thread, void* (*start_routine)(void*), void* arg) {
    return pthread_create(thread, NULL, start_routine, arg);
}

// Ожидание завершения потока
int ocean_thread_join(pthread_t thread) {
    return pthread_join(thread, NULL);
}

// Функция-диспетчер: принимает ocean_thread_task_t*, извлекает start_routine и arg,
// вызывает start_routine(arg) и возвращает результат.
void* ocean_thread_dispatch(void* task_ptr) {
    ocean_thread_task_t* task = (ocean_thread_task_t*)task_ptr;
    void* result = task->start_routine(task->arg);
    // Освобождаем структуру после завершения
    free(task);
    return result;
}

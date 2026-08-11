from tests.base import run


def test_c_code_math():
    P = r"""
cimport <math.h>

def main() -> float:
    unsafe:
        var a: float = @sqrt(16)   # C code -> function should starts with @
    return a
"""

    C = """
float main(void) {
    float a = sqrt(16);
    float ocean_return_0 = a;
    return ocean_return_0;
}
"""
    run(P, C)


def test_c_code_pthread():
    P = r"""
cimport <stdio.h>
cimport <stdlib.h>
cimport <string.h>
cimport <stdbool.h>
cimport <pthread.h>

class Object:
    def __init__(self, a: int):
        self.a = a
    
    def get_a(self) -> int:
        return self.a

def backward_worker(arg: None) -> None:
    var a: Object = arg
    var b: int = a.get_a()
    print(b)
    return None

def main() -> int:
    var thread: pthread_t = None
    var backward_thread_data: Object = Object(100)

    unsafe:
        @pthread_create(&thread, NULL, backward_worker, backward_thread_data)
        @pthread_join(thread, NULL)
    return 0
"""

    C = r"""
typedef struct ocean_Object ocean_Object;

struct ocean_Object {
    ocean_object_header header;
    void** vtable;
    int a;
};

int ocean_Object_get_a(ocean_Object* self);
int main(void);

static void ocean_destroy_Object(void* ptr) {
    ocean_Object* self = (ocean_Object*)ptr;
    if (!self) return;
    free(self);
}

ocean_Object* ocean_create_Object(int a) {
    ocean_Object* obj = (ocean_Object*)calloc(1, sizeof(ocean_Object));
    if (!obj) { fprintf(stderr, "Memory allocation failed for Object\n"); exit(1); }
    obj->header.refcount = 1;
    obj->header.destroy = ocean_destroy_Object;
    obj->vtable = NULL;
    obj->a = a;
    return obj;
}

int ocean_Object_get_a(ocean_Object* self) {
    int ocean_return_0 = self->a;
    return ocean_return_0;
}

void* ocean_backward_worker(void* arg) {
    ocean_Object* a = arg;
    ocean_retain(a);
    int b = ocean_Object_get_a(a);
    printf("%d\n", b);
    ocean_release(a);
    a = NULL;
    return NULL;
    ocean_release(a);
    a = NULL;
}

int main(void) {
    pthread_t thread = (pthread_t){0};
    ocean_Object* backward_thread_data = ocean_create_Object(100);
    pthread_create(&thread, NULL, ocean_backward_worker, backward_thread_data);
    pthread_join(thread, NULL);
    int ocean_return_1 = 0;
    ocean_release(backward_thread_data);
    backward_thread_data = NULL;
    return ocean_return_1;
    ocean_release(backward_thread_data);
    backward_thread_data = NULL;
}
"""
    run(P, C)

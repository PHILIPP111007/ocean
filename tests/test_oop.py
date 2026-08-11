from tests.base import run


def test_oop_1():
    P = r"""
class Object:
    def __init__(self, age: int) -> None:
        pass

class User(Object):
    def __init__(self, age: int, a: int) -> None:
        self.age = age
    
    def get_age(self) -> int:
        return self.age


def main() -> int:
    var u: User = User(10, 1)
    print(u.age)

    var age: int = u.get_age()
    print(age)

    return 0
"""

    C = r"""
typedef struct ocean_Object ocean_Object;

struct ocean_Object {
    ocean_object_header header;
    void** vtable;
};

typedef struct ocean_User ocean_User;

struct ocean_User {
    ocean_Object base;
    int age;
};

int ocean_User_get_age(ocean_User* self);
int main(void);

static void ocean_destroy_Object(void* ptr) {
    ocean_Object* self = (ocean_Object*)ptr;
    if (!self) return;
    free(self);
}

ocean_Object* ocean_create_Object(int age) {
    ocean_Object* obj = (ocean_Object*)calloc(1, sizeof(ocean_Object));
    if (!obj) { fprintf(stderr, "Memory allocation failed for Object\n"); exit(1); }
    obj->header.refcount = 1;
    obj->header.destroy = ocean_destroy_Object;
    obj->vtable = NULL;
    return obj;
}

static void ocean_destroy_User(void* ptr) {
    ocean_User* self = (ocean_User*)ptr;
    if (!self) return;
    free(self);
}

ocean_User* ocean_create_User(int age, int a) {
    ocean_User* obj = (ocean_User*)calloc(1, sizeof(ocean_User));
    if (!obj) { fprintf(stderr, "Memory allocation failed for User\n"); exit(1); }
    obj->base.header.refcount = 1;
    obj->base.header.destroy = ocean_destroy_User;
    obj->base.vtable = NULL;
    obj->age = age;
    return obj;
}

int ocean_User_get_age(ocean_User* self) {
    int ocean_return_0 = self->age;
    return ocean_return_0;
}

int main(void) {
    ocean_User* u = ocean_create_User(10, 1);
    printf("%d\n", u->age);
    int age = ocean_User_get_age(u);
    printf("%d\n", age);
    int ocean_return_1 = 0;
    ocean_release(u);
    u = NULL;
    return ocean_return_1;
    ocean_release(u);
    u = NULL;
}
"""
    run(P, C)


def disabled_test_oop_2():
    P = r"""
class A:
    def get_age_2(self) -> int:
        return 1

class B:
    def get_age(self) -> int:
        return 1
    
    def get_age_1(self) -> int:
        return 10

class User(A, B):
    def __init__(self, age: int, a: int) -> None:
        self.age = age
    
    def get_age(self) -> int:
        return self.age

def main() -> int:
    var u: User = User(10, 1)
    var age: int = u.get_age_2()

    print(age)

    return 0
"""

    C = r"""
"""
    run(P, C)


def test_oop_3():
    P = r"""
class Matrix:
    def __init__(self, data: list[int]):
        self.data = data
    
    def get(self) -> int:
        var item: int = self.data[10]
        return item
"""

    C = r"""
typedef struct ocean_Matrix ocean_Matrix;

struct ocean_Matrix {
    ocean_object_header header;
    void** vtable;
    ocean_list_int* data;
};

int ocean_Matrix_get(ocean_Matrix* self);
int main(void);

static void ocean_destroy_Matrix(void* ptr) {
    ocean_Matrix* self = (ocean_Matrix*)ptr;
    if (!self) return;
    ocean_release(((ocean_Matrix*)self)->data);
    free(self);
}

ocean_Matrix* ocean_create_Matrix(ocean_list_int* data) {
    ocean_Matrix* obj = (ocean_Matrix*)calloc(1, sizeof(ocean_Matrix));
    if (!obj) { fprintf(stderr, "Memory allocation failed for Matrix\n"); exit(1); }
    obj->header.refcount = 1;
    obj->header.destroy = ocean_destroy_Matrix;
    obj->vtable = NULL;
    ocean_retain(data);
    obj->data = data;
    return obj;
}

int ocean_Matrix_get(ocean_Matrix* self) {
    int item = ocean_get_list_int(self->data, 10);
    int ocean_return_0 = item;
    return ocean_return_0;
}
"""
    run(P, C)


def test_oop_4():
    P = r"""
class A:
    def __init__(self) -> None:
        self.value: int = 100

    def get_value(self) -> int:
        return self.value


class B:
    def __init__(self) -> None:
        pass

    def get_A_value(self) -> int:
        var a: A = A()
        var value: int = a.value
        value = a.get_value()
        return value

    def set_A_value(self, new_value: int) -> None:
        var a: A = A()
        a.value = new_value


class C(A):
    def __init__(self) -> None:
        pass        


def main() -> int:
    var c: C = C()
    print(c.get_value())
    return 0
"""

    C = r"""
typedef struct ocean_A ocean_A;

struct ocean_A {
    ocean_object_header header;
    void** vtable;
    int value;
};

typedef struct ocean_B ocean_B;

struct ocean_B {
    ocean_object_header header;
    void** vtable;
};

typedef struct ocean_C ocean_C;

struct ocean_C {
    ocean_A base;
};

int ocean_A_get_value(ocean_A* self);
int ocean_B_get_A_value(ocean_B* self);
void* ocean_B_set_A_value(ocean_B* self, int new_value);
int main(void);

static void ocean_destroy_A(void* ptr) {
    ocean_A* self = (ocean_A*)ptr;
    if (!self) return;
    free(self);
}

ocean_A* ocean_create_A(void) {
    ocean_A* obj = (ocean_A*)calloc(1, sizeof(ocean_A));
    if (!obj) { fprintf(stderr, "Memory allocation failed for A\n"); exit(1); }
    obj->header.refcount = 1;
    obj->header.destroy = ocean_destroy_A;
    obj->vtable = NULL;
    obj->value = 100;
    return obj;
}

static void ocean_destroy_B(void* ptr) {
    ocean_B* self = (ocean_B*)ptr;
    if (!self) return;
    free(self);
}

ocean_B* ocean_create_B(void) {
    ocean_B* obj = (ocean_B*)calloc(1, sizeof(ocean_B));
    if (!obj) { fprintf(stderr, "Memory allocation failed for B\n"); exit(1); }
    obj->header.refcount = 1;
    obj->header.destroy = ocean_destroy_B;
    obj->vtable = NULL;
    return obj;
}

static void ocean_destroy_C(void* ptr) {
    ocean_C* self = (ocean_C*)ptr;
    if (!self) return;
    free(self);
}

ocean_C* ocean_create_C(void) {
    ocean_C* obj = (ocean_C*)calloc(1, sizeof(ocean_C));
    if (!obj) { fprintf(stderr, "Memory allocation failed for C\n"); exit(1); }
    obj->base.header.refcount = 1;
    obj->base.header.destroy = ocean_destroy_C;
    obj->base.vtable = NULL;
    return obj;
}

int ocean_C_get_value(ocean_C* self) {
    // Вызов унаследованного метода из A
    ocean_A* base_obj = (ocean_A*)self;
    return ocean_A_get_value(base_obj);
}

int ocean_A_get_value(ocean_A* self) {
    int ocean_return_0 = self->value;
    return ocean_return_0;
}

int ocean_B_get_A_value(ocean_B* self) {
    ocean_A* a = ocean_create_A();
    int value = a->value;
    value = ocean_A_get_value(a);
    int ocean_return_1 = value;
    ocean_release(a);
    a = NULL;
    return ocean_return_1;
    ocean_release(a);
    a = NULL;
}

void* ocean_B_set_A_value(ocean_B* self, int new_value) {
    ocean_A* a = ocean_create_A();
    a->value = new_value;
    ocean_release(a);
    a = NULL;
}

int main(void) {
    ocean_C* c = ocean_create_C();
    printf("%d\n", ocean_C_get_value(c));
    int ocean_return_2 = 0;
    ocean_release(c);
    c = NULL;
    return ocean_return_2;
    ocean_release(c);
    c = NULL;
}
"""
    run(P, C)

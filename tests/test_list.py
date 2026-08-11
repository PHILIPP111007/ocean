from tests.base import run


def test_list_append():
    P = r"""
def main() -> int:
    var a: list[int] = [1, 2, 3]
    a.append(1)
    del a

    var b: list[list[int]] = []
    var b_1: list[int] = [1, 2, 3, 4]
    b.append(b_1)
    del b

    return 0
"""

    C = """
int main(void) {
    ocean_list_int* a = ocean_create_list_int(4);
    ocean_append_list_int(a, 1);
    ocean_append_list_int(a, 2);
    ocean_append_list_int(a, 3);
    ocean_append_list_int(a, 1);
    // del a
    ocean_release(a);
    a = NULL;
    ocean_list_list_int* b = ocean_create_list_list_int(4);
    ocean_list_int* b_1 = ocean_create_list_int(4);
    ocean_append_list_int(b_1, 1);
    ocean_append_list_int(b_1, 2);
    ocean_append_list_int(b_1, 3);
    ocean_append_list_int(b_1, 4);
    ocean_append_list_list_int(b, b_1);
    // del b
    ocean_release(b);
    b = NULL;
    int ocean_return_0 = 0;
    ocean_release(b_1);
    b_1 = NULL;
    return ocean_return_0;
    ocean_release(b_1);
    b_1 = NULL;
}
"""
    run(P, C)


def test_list_pop_1():
    P = r"""
def main() -> int:
    var a: list[int] = [1, 2, 3]
    
    var b: int = a.pop(0)
    var c: int = a.pop()

    return 0
"""

    C = r"""
int main(void) {
    ocean_list_int* a = ocean_create_list_int(4);
    ocean_append_list_int(a, 1);
    ocean_append_list_int(a, 2);
    ocean_append_list_int(a, 3);
    int b = ocean_pop_list_int(a, 0);
    int c = ocean_pop_list_int(a, -1);
    int ocean_return_0 = 0;
    ocean_release(a);
    a = NULL;
    return ocean_return_0;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)


def test_list_pop_2():
    P = r"""
def main() -> int:
    var a: list[int] = [1, 2, 3]
    
    var b: int = 0
    b = a.pop(0)

    print(b)

    return 0
"""

    C = r"""
int main(void) {
    ocean_list_int* a = ocean_create_list_int(4);
    ocean_append_list_int(a, 1);
    ocean_append_list_int(a, 2);
    ocean_append_list_int(a, 3);
    int b = 0;
    b = ocean_pop_list_int(a, 0);
    printf("%d\n", b);
    int ocean_return_0 = 0;
    ocean_release(a);
    a = NULL;
    return ocean_return_0;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)


def test_list_set_list_int():
    P = r"""
def main() -> int:
    var a: list[int] = [1, 2, 3]
    a[0] = 10
    return 0
"""

    C = r"""
int main(void) {
    ocean_list_int* a = ocean_create_list_int(4);
    ocean_append_list_int(a, 1);
    ocean_append_list_int(a, 2);
    ocean_append_list_int(a, 3);
    ocean_set_list_int(a, 0, 10);
    int ocean_return_0 = 0;
    ocean_release(a);
    a = NULL;
    return ocean_return_0;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)


def test_list_get_list_int():
    P = r"""
def main() -> int:
    var a: list[int] = [1, 2, 3]
    var a1: int = a[0]
    return 0
"""

    C = r"""
int main(void) {
    ocean_list_int* a = ocean_create_list_int(4);
    ocean_append_list_int(a, 1);
    ocean_append_list_int(a, 2);
    ocean_append_list_int(a, 3);
    int a1 = ocean_get_list_int(a, 0);
    int ocean_return_0 = 0;
    ocean_release(a);
    a = NULL;
    return ocean_return_0;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)


def test_list_get_list_str():
    P = r"""
def main() -> int:
    var a: list[str] = ["1", "2"]
    var a1: str = a[0]
    return 0
"""

    C = r"""
int main(void) {
    ocean_list_str* a = ocean_create_list_str(4);
    ocean_append_list_str(a, "1");
    ocean_append_list_str(a, "2");
    char* a1 = ocean_strdup(ocean_get_list_str(a, 0));
    int ocean_return_0 = 0;
    free(a1);
    a1 = NULL;
    ocean_release(a);
    a = NULL;
    return ocean_return_0;
    free(a1);
    a1 = NULL;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)


def test_list_set_list_str():
    P = r"""
def main() -> int:
    var a: list[str] = ["1", "2"]
    a[0] = "100"
    return 0
"""

    C = r"""
int main(void) {
    ocean_list_str* a = ocean_create_list_str(4);
    ocean_append_list_str(a, "1");
    ocean_append_list_str(a, "2");
    ocean_set_list_str(a, 0, "100");
    int ocean_return_0 = 0;
    ocean_release(a);
    a = NULL;
    return ocean_return_0;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)


def test_list_add_number_to_list_item():
    P = r"""
def main() -> int:
    var a: list[int] = [1, 2, 3]
    a[1] += 1
    return 0
"""

    C = r"""
int main(void) {
    ocean_list_int* a = ocean_create_list_int(4);
    ocean_append_list_int(a, 1);
    ocean_append_list_int(a, 2);
    ocean_append_list_int(a, 3);
    int temp_0 = ocean_get_list_int(a, 1);
    temp_0 += 1;
    ocean_set_list_int(a, 1, temp_0);
    int ocean_return_1 = 0;
    ocean_release(a);
    a = NULL;
    return ocean_return_1;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)


def test_list_indexes():
    P = r"""
def main() -> int:
    var A: list[list[list[int]]] = []
    var a: list[list[int]] = []
    var a1: list[int] = [1, 2, 3, 4, 5]

    a.append(a1)
    A.append(a)

    a1[0] = 100
    a[0][0] = 100
    A[0][0][0] = 100 + 1

    var b: int = A[0][0][0]
    print(b)

    b = A[0][0][0] + A[0][0][1]
    A[0][0][2] = A[0][0][0] + A[0][0][1]

    return 0
"""

    C = r"""
int main(void) {
    ocean_list_list_list_int* A = ocean_create_list_list_list_int(4);
    ocean_list_list_int* a = ocean_create_list_list_int(4);
    ocean_list_int* a1 = ocean_create_list_int(5);
    ocean_append_list_int(a1, 1);
    ocean_append_list_int(a1, 2);
    ocean_append_list_int(a1, 3);
    ocean_append_list_int(a1, 4);
    ocean_append_list_int(a1, 5);
    ocean_append_list_list_int(a, a1);
    ocean_append_list_list_list_int(A, a);
    ocean_set_list_int(a1, 0, 100);
    ocean_list_int* temp_0 = ocean_get_list_list_int(a, 0);
    ocean_set_list_int(temp_0, 0, 100);
    ocean_list_list_int* temp_1 = ocean_get_list_list_list_int(A, 0);
    ocean_list_int* temp_2 = ocean_get_list_list_int(temp_1, 0);
    ocean_set_list_int(temp_2, 0, (100 + 1));
    int b = ocean_get_list_int(ocean_get_list_list_int(ocean_get_list_list_list_int(A, 0), 0), 0);
    printf("%d\n", b);
    b = (ocean_get_list_int(ocean_get_list_list_int(ocean_get_list_list_list_int(A, 0), 0), 0) + ocean_get_list_int(ocean_get_list_list_int(ocean_get_list_list_list_int(A, 0), 0), 1));
    ocean_list_list_int* temp_3 = ocean_get_list_list_list_int(A, 0);
    ocean_list_int* temp_4 = ocean_get_list_list_int(temp_3, 0);
    ocean_set_list_int(temp_4, 2, (ocean_get_list_int(ocean_get_list_list_int(ocean_get_list_list_list_int(A, 0), 0), 0) + ocean_get_list_int(ocean_get_list_list_int(ocean_get_list_list_list_int(A, 0), 0), 1)));
    int ocean_return_5 = 0;
    ocean_release(a1);
    a1 = NULL;
    ocean_release(a);
    a = NULL;
    ocean_release(A);
    A = NULL;
    return ocean_return_5;
    ocean_release(a1);
    a1 = NULL;
    ocean_release(a);
    a = NULL;
    ocean_release(A);
    A = NULL;
}
"""
    run(P, C)

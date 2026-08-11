from tests.base import run


def test_functions_1():
    P = r"""
def func_1() -> None:
    var x: int = 0
    x = x + 1

def main() -> int:
    def func_2() -> int:
        return 1
    var x: int = 10
    if x < 10:
        if 1 < 10:
            func_1()
        else:
            func_2()
    return 0
"""

    C = r"""

void* ocean_func_1(void) {
    int x = 0;
    x = (x + 1);
}

int main(void) {
    int x = 10;
    if ((x < 10)) {
        if ((1 < 10)) {
            ocean_func_1();
        }
        else {
            ocean_func_2();
        }
    }
    int ocean_return_0 = 0;
    return ocean_return_0;
}

int ocean_func_2(void) {
    int ocean_return_1 = 1;
    return ocean_return_1;
}
"""
    run(P, C)


def test_functions_2():
    P = r"""
def func() -> str:
    return "qepfko"

def main() -> int:
    var a: int = 1 + 10
    var b: str = func()

    del b

    return 0
"""

    C = r"""
char* ocean_func(void) {
    char* ocean_return_0 = ocean_strdup("qepfko");
    return ocean_return_0;
}

int main(void) {
    int a = (1 + 10);
    char* b = ocean_func();
    // del b
    free(b);
    b = NULL;
    int ocean_return_1 = 0;
    return ocean_return_1;
}
"""
    run(P, C)

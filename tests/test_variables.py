from tests.base import run


def test_variables():
    P = r"""
def main() -> int:
    var a: int = 1000000
    var b: str = "ewpfkeof"

    var c: bool = True
    var c1: bool = False

    var d: list[int] = [1, 2, 3]
    var d1: list[str] = ["bbb", "aaa"]
    var d2: list[list[int]] = [d, d]

    var e: tuple[int] = (1, 2, 3, 4)
    var d3: list[tuple[int]] = [e, e, e]

    var f: None = None

    var g: bool = 1 < 10
    var g1: bool = 1 < 10 and 10 >= 100

    return 0
"""

    C = r"""
int main(void);

int main(void) {
    int a = 1000000;
    char* b = ocean_strdup("ewpfkeof");
    bool c = true;
    bool c1 = false;
    ocean_list_int* d = ocean_create_list_int(4);
    ocean_append_list_int(d, 1);
    ocean_append_list_int(d, 2);
    ocean_append_list_int(d, 3);
    ocean_list_str* d1 = ocean_create_list_str(4);
    ocean_append_list_str(d1, "bbb");
    ocean_append_list_str(d1, "aaa");
    ocean_list_list_int* d2 = ocean_create_list_list_int(4);
    ocean_append_list_list_int(d2, d);
    ocean_append_list_list_int(d2, d);
    int ocean_tuple_items_0[4] = {
        1,
        2,
        3,
        4
    };
    ocean_tuple_int* e = ocean_create_tuple_int(ocean_tuple_items_0, 4);
    ocean_list_tuple_int* d3 = ocean_create_list_tuple_int(4);
    ocean_append_list_tuple_int(d3, e);
    ocean_append_list_tuple_int(d3, e);
    ocean_append_list_tuple_int(d3, e);
    void* f = (void*){0};
    bool g = (1 < 10);
    bool g1 = ((1 < 10) && (10 >= 100));
    int ocean_return_1 = 0;
    ocean_release(d3);
    d3 = NULL;
    ocean_release(e);
    e = NULL;
    ocean_release(d2);
    d2 = NULL;
    ocean_release(d1);
    d1 = NULL;
    ocean_release(d);
    d = NULL;
    free(b);
    b = NULL;
    return ocean_return_1;
    ocean_release(d3);
    d3 = NULL;
    ocean_release(e);
    e = NULL;
    ocean_release(d2);
    d2 = NULL;
    ocean_release(d1);
    d1 = NULL;
    ocean_release(d);
    d = NULL;
    free(b);
    b = NULL;
}
"""
    run(P, C)

from tests.base import run


def test_del():
    P = r"""
def main() -> int:
    var a: int = 100
    var b: str = "kkofrkfor"
    var c: bool = False
    var d: list[int] = [1, 2, 3]
    var e: tuple[int] = (1, 2, 3)
    var f: list[list[int]] = [d, d]
    var g: list[tuple[int]] = [e, e, e]

    del a
    del b
    del c
    del d
    del e
    del f
    del g

    return 0
"""

    C = """
int main(void) {
    int a = 100;
    char* b = ocean_strdup("kkofrkfor");
    bool c = false;
    ocean_list_int* d = ocean_create_list_int(4);
    ocean_append_list_int(d, 1);
    ocean_append_list_int(d, 2);
    ocean_append_list_int(d, 3);
    int ocean_tuple_items_0[3] = {
        1,
        2,
        3
    };
    ocean_tuple_int* e = ocean_create_tuple_int(ocean_tuple_items_0, 3);
    ocean_list_list_int* f = ocean_create_list_list_int(4);
    ocean_append_list_list_int(f, d);
    ocean_append_list_list_int(f, d);
    ocean_list_tuple_int* g = ocean_create_list_tuple_int(4);
    ocean_append_list_tuple_int(g, e);
    ocean_append_list_tuple_int(g, e);
    ocean_append_list_tuple_int(g, e);
    // del a
    a = 0;
    // del b
    free(b);
    b = NULL;
    // del c
    c = false;
    // del d
    ocean_release(d);
    d = NULL;
    // del e
    ocean_release(e);
    e = NULL;
    // del f
    ocean_release(f);
    f = NULL;
    // del g
    ocean_release(g);
    g = NULL;
    int ocean_return_1 = 0;
    return ocean_return_1;
}
"""
    run(P, C)

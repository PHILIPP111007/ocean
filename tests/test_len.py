from tests.base import run


def test_dict():
    P = r"""
def main() -> int:
    var l: list[str] = ["1", "2"]

    var a: dict[str, str] = {"a": "b"}
    var a_len: int = len(a)

    var b: str = "abc"
    var b_len: int = len(b)

    var c: list[int] = [1, 2, 3]
    var c_len: int = len(c)

    return 0
"""

    C = r"""
int main(void) {
    ocean_list_str* l = ocean_create_list_str(4);
    ocean_append_list_str(l, "1");
    ocean_append_list_str(l, "2");
    ocean_dict_str_str* a = ocean_create_dict_str_str(16);
    ocean_set_dict_str_str(a, "a", "b");
    int a_len = ocean_len_dict_str_str(a);
    char* b = ocean_strdup("abc");
    int b_len = strlen(b);
    ocean_list_int* c = ocean_create_list_int(4);
    ocean_append_list_int(c, 1);
    ocean_append_list_int(c, 2);
    ocean_append_list_int(c, 3);
    int c_len = ocean_builtin_len_list_int(c);
    int ocean_return_0 = 0;
    ocean_release(c);
    c = NULL;
    free(b);
    b = NULL;
    ocean_release(a);
    a = NULL;
    ocean_release(l);
    l = NULL;
    return ocean_return_0;
    ocean_release(c);
    c = NULL;
    free(b);
    b = NULL;
    ocean_release(a);
    a = NULL;
    ocean_release(l);
    l = NULL;
}
"""
    run(P, C)

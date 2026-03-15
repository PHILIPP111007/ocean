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
    list_str* l = create_list_str(4);
    append_list_str(l, "1");
    append_list_str(l, "2");
    dict_str_str* a = create_dict_str_str(16);
    set_dict_str_str(a, "a", "b");
    int a_len = len_dict_str_str(a);
    char* b = "abc";
    int b_len = strlen(b);
    list_int* c = create_list_int(4);
    append_list_int(c, 1);
    append_list_int(c, 2);
    append_list_int(c, 3);
    int c_len = builtin_len_list_int(c);
    return 0;
}
"""
    run(P, C)

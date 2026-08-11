from tests.base import run


def test_methods():
    P = r"""
def main() -> int:
    var a: str = "Hello {}"
    a = a.upper()
    a = a.lower()
    a = a.format("world")
    var a_list: list[str] = a.split(" ")
    a_list.sort()

    var b: int = 100
    var b1: str = str(b)

    var c: str = "10"
    var c1: int = int(c)

    var d: list[int] = []
    d.append(1)
    d.append(2)
    d.sort()
    d.pop()

    var d1: list[list[int]] = []
    d1.append(d)
    d1.append(d)

    var e: int = len(d)
    var e1: int = len(d1)

    return 0
"""

    C = r"""
int main(void);

int main(void) {
    char* a = ocean_strdup("Hello {}");
    char* ocean_string_tmp_0 = ocean_string_upper(a);
    free(a);
    a = ocean_string_tmp_0;
    char* ocean_string_tmp_1 = ocean_string_lower(a);
    free(a);
    a = ocean_string_tmp_1;
    char* ocean_string_tmp_2 = ocean_string_format(a, "world");
    free(a);
    a = ocean_string_tmp_2;
    ocean_list_str* a_list = ocean_string_split(a, " ");
    qsort(a_list->data, a_list->size, sizeof(char*), ocean_compare_string);
    int b = 100;
    char* b1 = ocean_builtin_str_int(b);
    char* c = ocean_strdup("10");
    int c1 = atoi(c);
    ocean_list_int* d = ocean_create_list_int(4);
    ocean_append_list_int(d, 1);
    ocean_append_list_int(d, 2);
    qsort(d->data, d->size, sizeof(int), ocean_compare_int);
    ocean_list_list_int* d1 = ocean_create_list_list_int(4);
    ocean_append_list_list_int(d1, d);
    ocean_append_list_list_int(d1, d);
    int e = ocean_builtin_len_list_int(d);
    int e1 = ocean_builtin_len_list_list_int(d1);
    int ocean_return_3 = 0;
    ocean_release(d1);
    d1 = NULL;
    ocean_release(d);
    d = NULL;
    free(c);
    c = NULL;
    free(b1);
    b1 = NULL;
    ocean_release(a_list);
    a_list = NULL;
    free(a);
    a = NULL;
    return ocean_return_3;
    ocean_release(d1);
    d1 = NULL;
    ocean_release(d);
    d = NULL;
    free(c);
    c = NULL;
    free(b1);
    b1 = NULL;
    ocean_release(a_list);
    a_list = NULL;
    free(a);
    a = NULL;
}
"""
    run(P, C)

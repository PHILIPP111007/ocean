from tests.base import run


def test_dict():
    P = r"""
def main() -> int:
    var a: dict[str, int] = {"a": 1}
    var a1: dict[int, str] = {1: "a"}
    var a2: dict[str, str] = {"1": "a"}

    var a3: list[dict[str, int]] = []
    a3.append(a)

    a["b"] = 2
    a1[2] = "b"

    var keys: list[str] = a.keys()
    var values: list[int] = a.values()

    var len_keys: int = len(keys)

    for i in range(len_keys):
        var key: str = keys[i]
        print(a[key])

    del a
    del a1
    del a2
    del a3

    return 0
"""

    C = r"""
int main(void) {
    ocean_dict_str_int* a = ocean_create_dict_str_int(16);
    ocean_set_dict_str_int(a, "a", 1);
    ocean_dict_int_str* a1 = ocean_create_dict_int_str(16);
    ocean_set_dict_int_str(a1, 1, "a");
    ocean_dict_str_str* a2 = ocean_create_dict_str_str(16);
    ocean_set_dict_str_str(a2, "1", "a");
    ocean_list_dict_str_int* a3 = ocean_create_list_dict_str_int(4);
    ocean_append_list_dict_str_int(a3, a);
    ocean_set_dict_str_int(a, "b", 2);
    ocean_set_dict_int_str(a1, 2, "b");
    ocean_list_str* keys = ocean_keys_dict_str_int(a);
    ocean_list_int* values = ocean_values_dict_str_int(a);
    int len_keys = ocean_builtin_len_list_str(keys);
    for (int i = 0; ((1) > 0 ? i < len_keys : i > len_keys); i += 1) {
        char* key = ocean_strdup(ocean_get_list_str(keys, i));
        printf("%d\n", ocean_get_dict_str_int(a, key));
        free(key);
        key = NULL;
    }
    // del a
    ocean_release(a);
    a = NULL;
    // del a1
    ocean_release(a1);
    a1 = NULL;
    // del a2
    ocean_release(a2);
    a2 = NULL;
    // del a3
    ocean_release(a3);
    a3 = NULL;
    int ocean_return_0 = 0;
    ocean_release(values);
    values = NULL;
    ocean_release(keys);
    keys = NULL;
    return ocean_return_0;
    ocean_release(values);
    values = NULL;
    ocean_release(keys);
    keys = NULL;
}
"""
    run(P, C)


def test_dict_get():
    P = r"""
def main() -> int:
    var a: dict[str, str] = {"a": "1"}
    var a1: str = a.get("a")
    var a2: str = a.get("a", "")

    var b: dict[str, int] = {"a": 1}
    var b1: int = b.get("a")
    var b2: int = b.get("a", 100)

    return 0
"""

    C = r"""

int main(void) {
    ocean_dict_str_str* a = ocean_create_dict_str_str(16);
    ocean_set_dict_str_str(a, "a", "1");
    char* a1 = ocean_strdup(ocean_get_default_dict_str_str(a, "a", NULL));
    char* a2 = ocean_strdup(ocean_get_default_dict_str_str(a, "a", ""));
    ocean_dict_str_int* b = ocean_create_dict_str_int(16);
    ocean_set_dict_str_int(b, "a", 1);
    int b1 = ocean_get_default_dict_str_int(b, "a", 0);
    int b2 = ocean_get_default_dict_str_int(b, "a", 100);
    int ocean_return_0 = 0;
    ocean_release(b);
    b = NULL;
    free(a2);
    a2 = NULL;
    free(a1);
    a1 = NULL;
    ocean_release(a);
    a = NULL;
    return ocean_return_0;
    ocean_release(b);
    b = NULL;
    free(a2);
    a2 = NULL;
    free(a1);
    a1 = NULL;
    ocean_release(a);
    a = NULL;
}
"""
    run(P, C)

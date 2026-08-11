from tests.base import run


def test_print():
    P = r"""
def main() -> int:
    for i in range(10):
        print(i, i, i, end="\n", sep="_")

    return 0
"""

    C = r"""
int main(void) {
    for (int i = 0; ((1) > 0 ? i < 10 : i > 10); i += 1) {
        printf("%d_%d_%d\n", i, i, i);
    }
    int ocean_return_0 = 0;
    return ocean_return_0;
}
"""
    run(P, C)

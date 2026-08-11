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


def test_print_preserves_string_labels_containing_equals():
    P = r'''
def main() -> int:
    var prediction: float32 = 0.5
    var loss: float32 = 0.25
    print("prediction =", prediction)
    print("loss =", loss)

    return 0
'''

    C = r'''
int main(void) {
    float prediction = 0.5;
    float loss = 0.25;
    printf("%s %f\n", "prediction =", prediction);
    printf("%s %f\n", "loss =", loss);
    int ocean_return_0 = 0;
    return ocean_return_0;
}
'''
    run(P, C)

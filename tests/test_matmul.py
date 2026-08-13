from tests.base import run
from src.compiler import CCodeGenerator
from src.parser import Parser


def test_matmul_1():
    P = r"""
def get_empty_result_matrix(rows: int, cols: int) -> list[list[int]]:
    var C: list[list[int]] = []
    var zeros: list[int] = []
    
    # Создаем одну строку с нулями
    for j in range(cols):
        zeros.append(0)

    # Копируем эту строку для всех строк матрицы
    for i in range(rows):
        # Создаем копию строки zeros
        var row: list[int] = []
        for j in range(cols):
            row.append(zeros[j])
        C.append(row)
    
    return C


def matmul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    var rows_A: int = len(A)
    var cols_A_list: list[int] = A[0]
    var cols_A: int = len(cols_A_list)

    var cols_B_list: list[int] = B[0]
    var cols_B: int = len(cols_B_list)
    
    var C: list[list[int]] = get_empty_result_matrix(rows_A, cols_B)

    for i in range(rows_A):
        var A_row: list[int] = A[i]  # Кешируем строку A
        var C_row: list[int] = C[i]  # Кешируем строку результата
        
        for k in range(cols_A):
            var A_ik: int = A_row[k]  # Элемент A[i][k]
            var B_row: list[int] = B[k]  # Кешируем строку B
            
            for j in range(cols_B):
                C_row[j] = C_row[j] + A_ik * B_row[j]
    
    return C


def main() -> int:
    var size: int = 100
    var A: list[list[int]] = []
    var B: list[list[int]] = []

    # Инициализация матриц
    for i in range(size):
        var row_a: list[int] = []
        var row_b: list[int] = []
        for j in range(size):
            row_a.append(i + j)
            row_b.append(i + j)
        A.append(row_a)
        B.append(row_b)

    print("Matrix created")

    # Кешируем размеры
    var rows_A: int = len(A)
    var cols_B_list: list[int] = B[0]
    var cols_B: int = len(cols_B_list)
    

    # Основной цикл
    for _ in range(1000):
        var result: list[list[int]] = matmul(A, B)
        del result

    return 0
"""

    C = r"""
int main(void);

ocean_list_list_int* ocean_get_empty_result_matrix(int rows, int cols) {
    ocean_list_list_int* C = ocean_create_list_list_int(4);
    ocean_list_int* zeros = ocean_create_list_int(4);
    for (int j = 0; ((1) > 0 ? j < cols : j > cols); j += 1) {
        ocean_append_list_int(zeros, 0);
    }
    for (int i = 0; ((1) > 0 ? i < rows : i > rows); i += 1) {
        ocean_list_int* row = ocean_create_list_int(4);
        for (int j = 0; ((1) > 0 ? j < cols : j > cols); j += 1) {
            ocean_append_list_int(row, ocean_get_list_int(zeros, j));
        }
        ocean_append_list_list_int(C, row);
        ocean_release(row);
        row = NULL;
    }
    ocean_list_list_int* ocean_return_0 = C;
    ocean_release(zeros);
    zeros = NULL;
    return ocean_return_0;
    ocean_release(zeros);
    zeros = NULL;
    ocean_release(C);
    C = NULL;
}

ocean_list_list_int* ocean_matmul(ocean_list_list_int* A, ocean_list_list_int* B) {
    int rows_A = ocean_builtin_len_list_list_int(A);
    ocean_list_int* cols_A_list = ocean_get_list_list_int(A, 0);
    ocean_retain(cols_A_list);
    int cols_A = ocean_builtin_len_list_int(cols_A_list);
    ocean_list_int* cols_B_list = ocean_get_list_list_int(B, 0);
    ocean_retain(cols_B_list);
    int cols_B = ocean_builtin_len_list_int(cols_B_list);
    ocean_list_list_int* C = ocean_get_empty_result_matrix(rows_A, cols_B);
    for (int i = 0; ((1) > 0 ? i < rows_A : i > rows_A); i += 1) {
        ocean_list_int* A_row = ocean_get_list_list_int(A, i);
        ocean_retain(A_row);
        ocean_list_int* C_row = ocean_get_list_list_int(C, i);
        ocean_retain(C_row);
        for (int k = 0; ((1) > 0 ? k < cols_A : k > cols_A); k += 1) {
            int A_ik = ocean_get_list_int(A_row, k);
            ocean_list_int* B_row = ocean_get_list_list_int(B, k);
            ocean_retain(B_row);
            for (int j = 0; ((1) > 0 ? j < cols_B : j > cols_B); j += 1) {
                ocean_set_list_int(C_row, j, (ocean_get_list_int(C_row, j) + (A_ik * ocean_get_list_int(B_row, j))));
            }
            ocean_release(B_row);
            B_row = NULL;
        }
        ocean_release(C_row);
        C_row = NULL;
        ocean_release(A_row);
        A_row = NULL;
    }
    ocean_list_list_int* ocean_return_1 = C;
    ocean_release(cols_B_list);
    cols_B_list = NULL;
    ocean_release(cols_A_list);
    cols_A_list = NULL;
    return ocean_return_1;
    ocean_release(C);
    C = NULL;
    ocean_release(cols_B_list);
    cols_B_list = NULL;
    ocean_release(cols_A_list);
    cols_A_list = NULL;
}

int main(void) {
    int size = 100;
    ocean_list_list_int* A = ocean_create_list_list_int(4);
    ocean_list_list_int* B = ocean_create_list_list_int(4);
    for (int i = 0; ((1) > 0 ? i < size : i > size); i += 1) {
        ocean_list_int* row_a = ocean_create_list_int(4);
        ocean_list_int* row_b = ocean_create_list_int(4);
        for (int j = 0; ((1) > 0 ? j < size : j > size); j += 1) {
            ocean_append_list_int(row_a, (i + j));
            ocean_append_list_int(row_b, (i + j));
        }
        ocean_append_list_list_int(A, row_a);
        ocean_append_list_list_int(B, row_b);
        ocean_release(row_b);
        row_b = NULL;
        ocean_release(row_a);
        row_a = NULL;
    }
    printf("%s\n", "Matrix created");
    int rows_A = ocean_builtin_len_list_list_int(A);
    ocean_list_int* cols_B_list = ocean_get_list_list_int(B, 0);
    ocean_retain(cols_B_list);
    int cols_B = ocean_builtin_len_list_int(cols_B_list);
    for (int _ = 0; ((1) > 0 ? _ < 1000 : _ > 1000); _ += 1) {
        ocean_list_list_int* result = ocean_matmul(A, B);
        // del result
        ocean_release(result);
        result = NULL;
    }
    int ocean_return_2 = 0;
    ocean_release(cols_B_list);
    cols_B_list = NULL;
    ocean_release(B);
    B = NULL;
    ocean_release(A);
    A = NULL;
    return ocean_return_2;
    ocean_release(cols_B_list);
    cols_B_list = NULL;
    ocean_release(B);
    B = NULL;
    ocean_release(A);
    A = NULL;
}
"""
    run(P, C)

def test_matmul_2():
    P = r"""
class Matrix:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.data: list[int] = []

    def init_matrix(self) -> None:
        var size: int = self.size
        var data: list[int] = []

        for _ in range(size):
            data.append(0)
        
        self.data = data
    
    def get(self, i: int, j: int) -> int:
        return self.data[i * self.cols + j]

    def set(self, i: int, j: int, value: int) -> None:
        self.data[i * self.cols + j] = value


def matmul(A: Matrix, B: Matrix, C: Matrix) -> None:
    var rows_A: int = A.rows
    var cols_A: int = A.cols
    var cols_B: int = B.cols

    # Оптимизация 1: кешируем указатели на data для быстрого доступа
    var A_data: list[int] = A.data
    var B_data: list[int] = B.data
    var C_data: list[int] = C.data

    var cols_A_local: int = cols_A
    var cols_B_local: int = cols_B

    # Оптимизация 3: изменение порядка циклов для лучшей локальности кеша
    # Вместо i-k-j используем i-j-k с разверткой
    for i in range(rows_A):
        var row_offset_A: int = i * cols_A_local
        var row_offset_C: int = i * cols_B_local

        for j in range(cols_B_local):
            var sum_val: int = 0
            var offset_C: int = row_offset_C + j

            for k in range(cols_A_local):
                sum_val = sum_val + A_data[row_offset_A + k] * B_data[k * cols_B_local + j]

            C_data[offset_C] = sum_val


def main() -> int:
    var data: list[int] = []

    var rows: int = 100
    var cols: int = 100

    var A: Matrix = Matrix(rows, cols)
    var B: Matrix = Matrix(rows, cols)

    A.init_matrix()
    B.init_matrix()

    print("Matrix created")

    var C: Matrix = Matrix(rows, cols)
    C.init_matrix()

    # Основной цикл
    for _ in range(1000):
        matmul(A, B, C)

    return 0
"""

    C = r"""
typedef struct ocean_Matrix ocean_Matrix;

struct ocean_Matrix {
    ocean_object_header header;
    void** vtable;
    int rows;
    int cols;
    int size;
    ocean_list_int* data;
};

void* ocean_Matrix_init_matrix(ocean_Matrix* self);
int ocean_Matrix_get(ocean_Matrix* self, int i, int j);
void* ocean_Matrix_set(ocean_Matrix* self, int i, int j, int value);
int main(void);

static void ocean_destroy_Matrix(void* ptr) {
    ocean_Matrix* self = (ocean_Matrix*)ptr;
    if (!self) return;
    ocean_release(((ocean_Matrix*)self)->data);
    free(self);
}

ocean_Matrix* ocean_create_Matrix(int rows, int cols) {
    ocean_Matrix* obj = (ocean_Matrix*)calloc(1, sizeof(ocean_Matrix));
    if (!obj) { fprintf(stderr, "Memory allocation failed for Matrix\n"); exit(1); }
    obj->header.refcount = 1;
    obj->header.destroy = ocean_destroy_Matrix;
    obj->vtable = NULL;
    obj->rows = rows;
    obj->cols = cols;
    obj->size = rows * cols;
    return obj;
}

void* ocean_Matrix_init_matrix(ocean_Matrix* self) {
    int size = self->size;
    ocean_list_int* data = ocean_create_list_int(4);
    for (int _ = 0; ((1) > 0 ? _ < size : _ > size); _ += 1) {
        ocean_append_list_int(data, 0);
    }
    ocean_list_int* ocean_field_tmp_0 = data;
    ocean_retain(ocean_field_tmp_0);
    ocean_release(self->data);
    self->data = ocean_field_tmp_0;
    ocean_release(data);
    data = NULL;
}

int ocean_Matrix_get(ocean_Matrix* self, int i, int j) {
    int ocean_return_1 = ocean_get_list_int(self->data, ((i * self->cols) + j));
    return ocean_return_1;
}

void* ocean_Matrix_set(ocean_Matrix* self, int i, int j, int value) {
    ocean_set_list_int(self->data, ((i * self->cols) + j), value);
}

void* ocean_matmul(ocean_Matrix* A, ocean_Matrix* B, ocean_Matrix* C) {
    int rows_A = A->rows;
    int cols_A = A->cols;
    int cols_B = B->cols;
    ocean_list_int* A_data = A->data;
    ocean_retain(A_data);
    ocean_list_int* B_data = B->data;
    ocean_retain(B_data);
    ocean_list_int* C_data = C->data;
    ocean_retain(C_data);
    int cols_A_local = cols_A;
    int cols_B_local = cols_B;
    for (int i = 0; ((1) > 0 ? i < rows_A : i > rows_A); i += 1) {
        int row_offset_A = (i * cols_A_local);
        int row_offset_C = (i * cols_B_local);
        for (int j = 0; ((1) > 0 ? j < cols_B_local : j > cols_B_local); j += 1) {
            int sum_val = 0;
            int offset_C = (row_offset_C + j);
            for (int k = 0; ((1) > 0 ? k < cols_A_local : k > cols_A_local); k += 1) {
                sum_val = (sum_val + (ocean_get_list_int(A_data, (row_offset_A + k)) * ocean_get_list_int(B_data, ((k * cols_B_local) + j))));
            }
            ocean_set_list_int(C_data, offset_C, sum_val);
        }
    }
    ocean_release(C_data);
    C_data = NULL;
    ocean_release(B_data);
    B_data = NULL;
    ocean_release(A_data);
    A_data = NULL;
}

int main(void) {
    ocean_list_int* data = ocean_create_list_int(4);
    int rows = 100;
    int cols = 100;
    ocean_Matrix* A = ocean_create_Matrix(rows, cols);
    ocean_Matrix* B = ocean_create_Matrix(rows, cols);
    ocean_Matrix_init_matrix(A);
    ocean_Matrix_init_matrix(B);
    printf("%s\n", "Matrix created");
    ocean_Matrix* C = ocean_create_Matrix(rows, cols);
    ocean_Matrix_init_matrix(C);
    for (int _ = 0; ((1) > 0 ? _ < 1000 : _ > 1000); _ += 1) {
        ocean_matmul(A, B, C);
    }
    int ocean_return_2 = 0;
    ocean_release(C);
    C = NULL;
    ocean_release(B);
    B = NULL;
    ocean_release(A);
    A = NULL;
    ocean_release(data);
    data = NULL;
    return ocean_return_2;
    ocean_release(C);
    C = NULL;
    ocean_release(B);
    B = NULL;
    ocean_release(A);
    A = NULL;
    ocean_release(data);
    data = NULL;
}
"""
    run(P, C)


def test_matmul_3():
    P = r"""
def matmul(A: list[int], B: list[int], C: list[int], rows_A: int, cols_A: int, cols_B: int) -> None:
    # Оптимизация 1: Изменение порядка циклов (i-j-k) для лучшей локальности кеша
    for i in range(rows_A):
        var offset_A: int = i * cols_A
        var offset_C: int = i * cols_B

        for j in range(cols_B):
            var sum_val: int = 0
            var offset_B: int = j  # Начальное смещение для j-го столбца B

            # Оптимизация 2: Ручная развертка цикла k (по 4 элемента)
            var k: int = 0
            var cols_A_local: int = cols_A
            
            while k + 3 < cols_A_local:
                sum_val = sum_val + A[offset_A + k] * B[k * cols_B + j]
                sum_val = sum_val + A[offset_A + k + 1] * B[(k + 1) * cols_B + j]
                sum_val = sum_val + A[offset_A + k + 2] * B[(k + 2) * cols_B + j]
                sum_val = sum_val + A[offset_A + k + 3] * B[(k + 3) * cols_B + j]
                k = k + 4

            # Обработка остатка
            while k < cols_A_local:
                sum_val = sum_val + A[offset_A + k] * B[k * cols_B + j]
                k = k + 1

            C[offset_C + j] = sum_val


def main() -> int:
    var size: int = 100
    var rows: int = size
    var cols: int = size
    var total_size: int = rows * cols

    # Создаем одномерные массивы для матриц A и B
    var A: list[int] = []
    var B: list[int] = []

    # Инициализация матриц
    for idx in range(total_size):
        A.append(0)
        B.append(0)

    print("Matrix created")

    # Создаем массив для результата C
    var C: list[int] = []
    for _ in range(rows * cols):
        C.append(0)

    # Основной цикл
    for _ in range(1000):
        matmul(A, B, C, rows, cols, cols)

    return 0
"""

    C = r"""
int main(void);

void* ocean_matmul(ocean_list_int* A, ocean_list_int* B, ocean_list_int* C, int rows_A, int cols_A, int cols_B) {
    for (int i = 0; ((1) > 0 ? i < rows_A : i > rows_A); i += 1) {
        int offset_A = (i * cols_A);
        int offset_C = (i * cols_B);
        for (int j = 0; ((1) > 0 ? j < cols_B : j > cols_B); j += 1) {
            int sum_val = 0;
            int offset_B = j;
            int k = 0;
            int cols_A_local = cols_A;
            while (((k + 3) < cols_A_local)) {
                sum_val = (sum_val + (ocean_get_list_int(A, (offset_A + k)) * ocean_get_list_int(B, ((k * cols_B) + j))));
                sum_val = (sum_val + (ocean_get_list_int(A, (offset_A + (k + 1))) * ocean_get_list_int(B, (((k + 1) * cols_B) + j))));
                sum_val = (sum_val + (ocean_get_list_int(A, (offset_A + (k + 2))) * ocean_get_list_int(B, (((k + 2) * cols_B) + j))));
                sum_val = (sum_val + (ocean_get_list_int(A, (offset_A + (k + 3))) * ocean_get_list_int(B, (((k + 3) * cols_B) + j))));
                k = (k + 4);
            }
            while ((k < cols_A_local)) {
                sum_val = (sum_val + (ocean_get_list_int(A, (offset_A + k)) * ocean_get_list_int(B, ((k * cols_B) + j))));
                k = (k + 1);
            }
            ocean_set_list_int(C, (offset_C + j), sum_val);
        }
    }
}

int main(void) {
    int size = 100;
    int rows = size;
    int cols = size;
    int total_size = (rows * cols);
    ocean_list_int* A = ocean_create_list_int(4);
    ocean_list_int* B = ocean_create_list_int(4);
    for (int idx = 0; ((1) > 0 ? idx < total_size : idx > total_size); idx += 1) {
        ocean_append_list_int(A, 0);
        ocean_append_list_int(B, 0);
    }
    printf("%s\n", "Matrix created");
    ocean_list_int* C = ocean_create_list_int(4);
    for (int _ = 0; ((1) > 0 ? _ < rows * cols : _ > rows * cols); _ += 1) {
        ocean_append_list_int(C, 0);
    }
    for (int _ = 0; ((1) > 0 ? _ < 1000 : _ > 1000); _ += 1) {
        ocean_matmul(A, B, C, rows, cols, cols);
    }
    int ocean_return_0 = 0;
    ocean_release(C);
    C = NULL;
    ocean_release(B);
    B = NULL;
    ocean_release(A);
    A = NULL;
    return ocean_return_0;
    ocean_release(C);
    C = NULL;
    ocean_release(B);
    B = NULL;
    ocean_release(A);
    A = NULL;
}
"""
    run(P, C)

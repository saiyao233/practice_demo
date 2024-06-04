#include <iostream>
#include <omp.h>

const int N = 1000; // 矩阵大小

void matrix_multiply_Parallel(int **a, int **b, int **c) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            c[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
void matrix_multiply(int **a, int **b, int **c) {
    // #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            c[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    // 分配内存并初始化矩阵
    omp_set_num_threads(12);   //设置线程数
    int **a = new int*[N];
    int **b = new int*[N];
    int **c = new int*[N];
    for (int i = 0; i < N; ++i) {
        a[i] = new int[N];
        b[i] = new int[N];
        c[i] = new int[N];
        for (int j = 0; j < N; ++j) {
            a[i][j] = rand() % 100;
            b[i][j] = rand() % 100;
        }
    }

    // 测量优化前的运行时间
    double start_time = omp_get_wtime();
    matrix_multiply_Parallel(a, b, c);
    double end_time = omp_get_wtime();
    std::cout << "运行时间（优化）: " << end_time - start_time << " 秒" << std::endl;
    start_time = omp_get_wtime();
    matrix_multiply(a, b, c);
    end_time = omp_get_wtime();
    std::cout << "运行时间（未优化）: " << end_time - start_time << " 秒" << std::endl;
    // 释放内存
    for (int i = 0; i < N; ++i) {
        delete[] a[i];
        delete[] b[i];
        delete[] c[i];
    }
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

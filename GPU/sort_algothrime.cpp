#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

// 快速排序算法
void quickSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int pivot = arr[left];
        int low = left + 1;
        int high = right;
        while (true) {
            while (low <= high && arr[high] >= pivot) --high;
            while (low <= high && arr[low] <= pivot) ++low;
            if (low <= high) {
                std::swap(arr[low], arr[high]);
            } else {
                break;
            }
        }
        std::swap(arr[left], arr[high]);
        quickSort(arr, left, high - 1);
        quickSort(arr, high + 1, right);
    }
}

int main() {
    const int size = 1000000; // 数组大小
    std::vector<int> arr(size);

    // 初始化随机数组
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 1000;
    }

    // 测量优化前的运行时间
    double start_time = omp_get_wtime();

    // 调用快速排序算法
    quickSort(arr, 0, size - 1);

    double end_time = omp_get_wtime();
    std::cout << "运行时间（未优化）: " << end_time - start_time << " 秒" << std::endl;

    // 测量优化后的运行时间
    start_time = omp_get_wtime();

    // 使用 OpenMP 并行化快速排序算法
    #pragma omp parallel
    {
        #pragma omp single nowait
        quickSort(arr, 0, size - 1);
    }

    end_time = omp_get_wtime();
    std::cout << "运行时间（优化后）: " << end_time - start_time << " 秒" << std::endl;

    return 0;
}

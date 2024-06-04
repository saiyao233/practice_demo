#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

const int N = 1024; // 图像大小
const int BATCH = 1; // 批处理大小

// CUDA FFT 执行函数
void fft_cuda(cufftComplex* data, int width, int height) {
    cufftHandle plan;
    // 创建 FFT 计划
    cufftPlan2d(&plan, width, height, CUFFT_C2C);

    // 执行 FFT
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);

    // 销毁 FFT 计划
    cufftDestroy(plan);
}

int main() {
    // 分配内存并初始化数据
    cufftComplex* data;
    cudaMallocManaged(&data, N * N * BATCH * sizeof(cufftComplex));
    for (int i = 0; i < N * N * BATCH; ++i) {
        data[i].x = sin(2 * M_PI * i / (N * N));
        data[i].y = 0.0f;
    }

    // 测量优化前的运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 执行 FFT
    fft_cuda(data, N, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "运行时间（优化）: " << milliseconds << " 毫秒" << std::endl;

    // 释放内存
    cudaFree(data);

    return 0;
}

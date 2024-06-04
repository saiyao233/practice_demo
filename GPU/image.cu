#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void binaryThresholdKernel(unsigned char* input, unsigned char* output, int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        output[index] = (input[index] > threshold) ? 255 : 0;
    }
}

void binaryThreshold(unsigned char* input, unsigned char* output, int width, int height, unsigned char threshold) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    unsigned char* d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    binaryThresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, threshold);

    cudaMemcpy(output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // 读取图像
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "无法读取图像文件" << std::endl;
        return 1;
    }

    int width = image.cols;
    int height = image.rows;

    // 分配内存并拷贝图像数据到主机内存
    unsigned char* input = image.data;
    unsigned char* output = new unsigned char[width * height];

    // 定义阈值
    unsigned char threshold = 128;

    // 测量运行时间
    double start_time = cv::getTickCount();

    // 调用二值化函数
    binaryThreshold(input, output, width, height, threshold);

    double end_time = cv::getTickCount();
    double elapsed_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "运行时间: " << elapsed_time << " 秒" << std::endl;

    // 创建二值化后的图像
    cv::Mat binary_image(height, width, CV_8UC1, output);

    // 保存二值化后的图像
    cv::imwrite("output_binary.jpg", binary_image);

    std::cout << "二值化后的图像已保存为 output_binary.jpg" << std::endl;

    // 释放内存
    delete[] output;

    return 0;
}
// nvcc your_program.cu -o your_program `pkg-config --cflags --libs opencv` -lopencv_core -lopencv_imgcodecs

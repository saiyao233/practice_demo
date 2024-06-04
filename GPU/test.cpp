//test_opencv.cpp
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
//! [includes]

int main()
{
    //! [imread]
    std::string image_path = "input.jpg";//samples::findFile("starry_night.jpg")"";
    Mat img = imread(image_path, IMREAD_COLOR);
    //! [imread]

    //! [empty]
    cv::imwrite("output.jpg", img);

    std::cout << "图像处理完成，已保存为 output.jpg" << std::endl;

    return 0;
}
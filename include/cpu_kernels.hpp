//
// Created by Matt on 2019-12-31.
//
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <omp.h>
#include "helper.hpp"

#ifndef CUDA_CPU_KERNELS_HPP
#define CUDA_CPU_KERNELS_HPP

class cpuKernel {
private:

public:
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma);
    cv::Mat conv_image(const cv::Mat &frame, std::vector<std::vector<float>> &matrix);
};

#endif //CUDA_CPU_KERNELS_HPP

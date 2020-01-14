//
// Created by Matt on 2019-12-31.
//
#include <opencv2/imgproc/imgproc.hpp>

#if defined(cuda_compile_check)
#include "cuda_kernels.hpp"
#elif defined(opencl_compile_check)
#include "opencl_kernels.hpp"
#else
#include "../include/cpu_kernels.hpp"
#endif

#ifndef CUDA_ACCELERATOR_HPP
#define CUDA_ACCELERATOR_HPP

class Accelerator {
private:
#if defined(cuda_compile_check)
    cudaKernel dev;
#elif defined(opencl_compile_check)
    openCLKernel dev;
#else
    cpuKernel dev;
#endif
public:
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma) {return dev.gaussian_blur(frame, kernelSize, sigma);};
    cv::Mat conv(const cv::Mat &frame, std::vector<std::vector<float>> &matrix) {return dev.conv_image(frame, matrix);};
};

#endif //CUDA_ACCELERATOR_HPP

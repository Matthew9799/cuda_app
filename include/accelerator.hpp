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
public:
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma);
};

#endif //CUDA_ACCELERATOR_HPP

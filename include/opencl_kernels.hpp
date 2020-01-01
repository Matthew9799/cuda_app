//
// Created by Matt on 2019-12-14.
//
//
#include <opencv2/imgproc/imgproc.hpp>

#ifndef CUDA_OPENCL_KERNELS_HPP
#define CUDA_OPENCL_KERNELS_HPP

class openCLKernel
{
public:
    openCLKernel();
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma);
};

#endif //CUDA_OPENCL_KERNELS_HPP

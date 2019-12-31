//
// Created by Matt on 2019-12-31.
//
#include <opencv2/imgproc/imgproc.hpp>
#include "cuda_kernels.hpp"
#include "opencl_kernels.hpp"

#ifndef CUDA_ACCELERATOR_HPP
#define CUDA_ACCELERATOR_HPP

class Accelerator {
private:
    cudaKernel *cuda_gpu;
    openCLKernel *openCLDevice;
public:
    Accelerator();
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma);
};

#endif //CUDA_ACCELERATOR_HPP

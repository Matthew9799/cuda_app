//
// Created by Matt on 2019-12-31.
//
#include <opencv2/imgproc/imgproc.hpp>

#ifndef CUDA_CPU_KERNELS_HPP
#define CUDA_CPU_KERNELS_HPP

class cpuKernel {
private:

public:
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma);
};

#endif //CUDA_CPU_KERNELS_HPP

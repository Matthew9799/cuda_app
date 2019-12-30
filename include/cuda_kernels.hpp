//
// Created by Matt on 2019-12-14.
//
#include <opencv2/imgproc/imgproc.hpp>

#ifndef CUDA_CUDA_KERNELS_HPP
#define CUDA_CUDA_KERNELS_HPP

class cudaKernel
{
private:

public:
    static cv::Mat gaussian_blur(cv::Mat frame, float intensity);
};

#endif //CUDA_OPENCL_KERNELS_HPP
//
// Created by Matt on 2019-12-14.
//
#include <opencv2/imgproc/imgproc.hpp>
#include "helper.hpp"
#include <omp.h>
#include <thread>

#ifndef CUDA_CUDA_KERNELS_HPP
#define CUDA_CUDA_KERNELS_HPP

class cudaKernel
{
private:
    uint8_t streamingProcs;
    uint8_t maxThreadsPerBlock;
    size_t totalMem;
    size_t sharedMemPerBlock;
    uint8_t regPerBlock;
    uint8_t concurrentCopy;
    char * name;
public:
    cudaKernel();
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma);
};

#endif //CUDA_OPENCL_KERNELS_HPP
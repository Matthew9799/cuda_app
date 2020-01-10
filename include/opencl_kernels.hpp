//
// Created by Matt on 2019-12-14.
//
//
#include <opencv2/imgproc/imgproc.hpp>
#include "helper.hpp"
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <omp.h>
#include <thread>
#include <string>

#ifndef CUDA_OPENCL_KERNELS_HPP
#define CUDA_OPENCL_KERNELS_HPP

class openCLKernel
{
private:
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
public:
    openCLKernel();
    ~openCLKernel();
    cv::Mat gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma);
};

#endif //CUDA_OPENCL_KERNELS_HPP

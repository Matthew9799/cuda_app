#include "../include/opencl_kernels.hpp"
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

openCLKernel::openCLKernel() {

}


cv::Mat openCLKernel::gaussian_blur(const cv::Mat & frame, int kernelSize, float sigma) {

    return cv::Mat();
}


#include "../include/accelerator.hpp"

/*
 * Performs Gaussian blur based on the proper device initialized
 */

cv::Mat Accelerator::gaussian_blur(const cv::Mat & frame, int kernelSize, float sigma) {
    cv::Mat result;
#if defined(cuda_compile_check)
    cudaKernel dev;
    result = dev.gaussian_blur(frame, kernelSize, sigma);
#elif defined(opencl_compile_check)
    openCLKernel dev;
    frame = dev.gaussian_blur(frame, kernelSize, sigma);
#else
    cpuKernel dev;
    result = dev.gaussian_blur(frame, kernelSize, sigma);
#endif
    return result;
}
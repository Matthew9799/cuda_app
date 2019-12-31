#include "../include/accelerator.hpp"

/*
 * Constructor will initialize with the proper device based on the hardware present in the clients computer
 */
Accelerator::Accelerator() {
    if(cudaKernel::test_device()) {
        cuda_gpu = new cudaKernel();
        openCLDevice = NULL;
    } else if(openCLKernel::test_device()){
        openCLDevice = new openCLKernel();
        cuda_gpu = NULL;
    }
}

/*
 * Performs Gaussian blur based on the proper device initialized
 */

cv::Mat Accelerator::gaussian_blur(const cv::Mat & frame, int kernelSize, float sigma) {
    cv::Mat result;
    if(cuda_gpu) {
        result = cuda_gpu->gaussian_blur(frame,kernelSize,sigma);
    } else {
        result = openCLDevice->gaussian_blur(frame, kernelSize,sigma);
    }
    return result;
}
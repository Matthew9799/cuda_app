#include "../include/cuda_kernels.hpp"

/*
 * Function takes CV Mat, and intensity and sets up the call so that a CUDA device can accelerate the task.
 * Function then cleans up and returns the result.
 */
static cv::Mat gaussian_blur(cv::Mat frame, float intensity) {
    cv::Mat dat = frame.clone();
    return dat;
}

__global__ void cuda_blur(cv::Mat fame, float intensity) {

}
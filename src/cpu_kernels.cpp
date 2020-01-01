//
// Created by Matt on 2019-12-31.
//
#include "../include/cpu_kernels.hpp"

cv::Mat cpuKernel::gaussian_blur(const cv::Mat & frame, int kernelSize, float sigma) {
    cv::Mat data(cv::Size(frame.rows, frame.cols), CV_8UC3);
    float *conv = (float*)malloc(kernelSize*kernelSize*sizeof(float));
    unsigned int nthreads = std::thread::hardware_concurrency();

    if(kernelSize > 1 && kernelSize%2) {
        helper::gaussian_convolution(conv, kernelSize, sigma);
    } else if(kernelSize == 1 && kernelSize%2) {
        conv[0] = 1;
    } else {
        delete(conv);
        throw "Kernel size cannot be less than one and kernel size must be odd";
    }

    omp_set_num_threads(nthreads); // set threads to all machine has

    #pragma omp for
    for(int y = 0; y < frame.rows; y++) { // loop through rows
        for(int x = 0; x < frame.cols; x++) { // loop through columns

            for(int y1 = -kernelSize/2; y1 <= kernelSize/2; y1++) { // loop through y val of conv matrix
                for(int x1 = -kernelSize/2; x1 <= kernelSize/2; x1++) { // loop through x val of conv matrix

                    if(y+y1 >= 0 && y+y1 < frame.rows) { // check to see if out of bounds of rows
                        if(x+x1 >= 0 && x+x1 < frame.cols) {
                            data.at<cv::Vec3b>(y,x)[0] = frame.at<cv::Vec3b>(y,x)[0]*conv[(kernelSize/2+y1)*kernelSize+(kernelSize/2+x1)]; // B
                            data.at<cv::Vec3b>(y,x)[1] = frame.at<cv::Vec3b>(y,x)[1]*conv[(kernelSize/2+y1)*kernelSize+(kernelSize/2+x1)]; // G
                            data.at<cv::Vec3b>(y,x)[2] = frame.at<cv::Vec3b>(y,x)[2]*conv[(kernelSize/2+y1)*kernelSize+(kernelSize/2+x1)]; // R
                        }
                    }
                }
            }
        }
    }

    delete(conv);

    return data;
}


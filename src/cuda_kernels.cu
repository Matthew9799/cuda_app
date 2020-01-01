#include "../include/cuda_kernels.hpp"

/*
 * Class constructor to retrieve all relevant cuda variables. May be removed later, but figured it might be
 * useful to have this information when deciding on block or thread counts.
 */

cudaKernel::cudaKernel() {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    streamingProcs = devProp.multiProcessorCount;
    maxThreadsPerBlock = devProp.maxThreadsPerBlock;
    totalMem = devProp.totalGlobalMem;
    sharedMemPerBlock = devProp.sharedMemPerBlock;
    regPerBlock = devProp.regsPerBlock;
    concurrentCopy = (devProp.deviceOverlap ? 1 : 0);
    name = devProp.name;
}

/*
 * Checks to see if an error has occcured
 */

void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); // bail out immediately (makes debugging easier)
    }
}

/*
 * Function is passed the image as represented by an array of unsigned chars. Data is then written into the second
 * parameter and then retrieved by caller.
 */

__global__ void cuda_gaussian_blur(const uchar *image, uchar *returnImage, const uint64 length, int kernelSize, float * conv) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x; // have pixel working on
}

/*
 * Function takes CV Mat, and intensity and sets up the call so that a CUDA device can accelerate the task.
 * Function then cleans up and returns the result.
 */

cv::Mat cudaKernel::gaussian_blur(const cv::Mat &frame, int kernelSize, float sigma) {
    cv::Mat finalResult; // return data
    cudaError_t statusA; // error of initialization
    uchar *dev_imageA; // gpu pointer for image
    uchar *dev_imageB; // gpu pointer to return image
    std::vector<uchar> *array = new std::vector<uchar>; // frame into vector
    std::vector<uchar> *results; // vector into frame
    uchar *host_image; // array for frame
    float *conv = (float*)malloc(kernelSize*kernelSize*sizeof(float));
    float *dev_conv;

    if (frame.isContinuous()) {
        array->assign(frame.data, frame.data + frame.total());
    } else {
        for (int i = 0; i < frame.rows; ++i) {
            array->insert(array->end(), frame.ptr<uchar>(i), frame.ptr<uchar>(i) + frame.cols);
        }
    }

    if(kernelSize > 1 && kernelSize%2) {
        helper::gaussian_convolution(conv, kernelSize, sigma);
    } else if(kernelSize == 1 && kernelSize%2) {
        conv[0] = 1;
    } else {
        delete(conv);
        throw "Kernel size cannot be less than one and kernel size must be odd";
    }

    host_image = array->data(); // convert image vector to prep array to copy to GPU
    statusA = cudaMalloc(&dev_imageA, array->size()*sizeof(uchar)); // allocate mem on gpu
    check_error(statusA, "Failed Allocation 1");
    statusA = cudaMalloc(&dev_imageB, array->size()* sizeof(uchar)); // allocate mem on gpu
    check_error(statusA, "Failed Allocation 2");
    statusA = cudaMalloc(&dev_conv, kernelSize*kernelSize* sizeof(float));
    check_error(statusA, "Failed Allocation 3");

    statusA = cudaMemcpy(dev_imageA, host_image, array->size()*sizeof(uchar), cudaMemcpyHostToDevice);
    check_error(statusA, "Failed cuda host to device copy 1");
    statusA = cudaMemcpy(dev_conv, conv, kernelSize*kernelSize*sizeof(float), cudaMemcpyHostToDevice);
    check_error(statusA, "Failed cuda host to device copy 2");

    uint32_t numBlocks = (int)ceilf((float)(frame.rows*frame.cols)/1024);

    cuda_gaussian_blur<<<numBlocks, 1024>>>(dev_imageA, dev_imageB, array->size()*sizeof(uchar), kernelSize, dev_conv);
    check_error(cudaGetLastError(), "Error in kernel.");

    statusA = cudaMemcpy(host_image, dev_imageA, array->size()* sizeof(uchar), cudaMemcpyDeviceToHost);
    check_error(statusA, "Failed cuda device to host copy");

    results = new std::vector<uchar>(host_image, host_image+array->size());
    finalResult = cv::Mat(frame.rows, frame.cols, frame.type(), results->data());

    statusA = cudaFree(dev_imageA);
    check_error(statusA, "Failed to free device memory 1");
    statusA = cudaFree(dev_imageB);
    check_error(statusA, "Failed to free device memory 2");

    delete(results);
    delete(array);
    delete(conv);

    return finalResult;
}
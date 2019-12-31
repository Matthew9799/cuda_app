#include "../include/opencl_kernels.hpp"
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

openCLKernel::openCLKernel() {

}

#define MAX_PLATFORM_SIZE 256
#define MAX_DEVICE_SIZE 256

int openCLKernel::test_device() {
    cl_device_id devices[256];
    cl_platform_id platforms[256];
    cl_uint numDevices;
    cl_uint numPlatforms;
    cl_uint ret;
    cl_uint i, j;
    char buf[4096];


    ret = clGetPlatformIDs(MAX_PLATFORM_SIZE, platforms, &numPlatforms);
    if (ret != CL_SUCCESS) {
        exit(-1);
    }

    for (i=0; i<numPlatforms; i++) {

        ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        if (ret != CL_SUCCESS) {
            exit(-1);
        }

        printf("Platform[%d] : %s\n\n", i, buf);

        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICE_SIZE, devices, &numDevices);
        if (ret != CL_SUCCESS) {
            exit(-1);
        }

        for (j=0; j<numDevices; j++) {
            printf("--- Device[%d] ---\n", j);
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
            if (ret != CL_SUCCESS) {
                exit(-1);
            }
            printf("Device Name : %s\n", buf);

            ret = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buf), buf, NULL);
            if (ret != CL_SUCCESS) {
                exit(-1);
            }
            printf("Device Vendor : %s\n", buf);

            printf("------\n");

        }
        printf("\n");

    }
}

cv::Mat openCLKernel::gaussian_blur(const cv::Mat & frame, int kernelSize, float sigma) {
    return cv::Mat();
}


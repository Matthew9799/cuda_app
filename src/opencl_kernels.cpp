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
    cv::Mat finalResult = cv::Mat(cv::Size(frame.cols, frame.rows), CV_8UC3);; // return data
    std::vector<uchar> *array = new std::vector<uchar>(); // frame into vector
    uchar *host_image; // array for frame
    float *conv = (float*)malloc(kernelSize*kernelSize*sizeof(float));
    uint64 length = frame.rows * frame.cols * frame.channels();
    size_t globalSize, localSize;
    FILE* programHandle;
    size_t programSize, kernelSourceSize;
    char *programBuffer, *kernelSource;

    cl_mem dev_image;   // Storage for input image
    cl_mem dev_result;  // Storage for output image
    cl_mem dev_conv;    // Storage for conv matrix

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
    cl_int err;                       // error flag

    if(kernelSize > 1 && kernelSize%2) {
        helper::gaussian_convolution(conv, kernelSize, sigma);
    } else if(kernelSize == 1 && kernelSize%2) {
        conv[0] = 1;
    } else {
        delete(conv);
        delete(array);
        throw "Kernel size cannot be less than one and kernel size must be odd";
    }

    for(int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            cv::Vec3b temp = frame.at<cv::Vec3b>(y, x);
            array->push_back(temp[0]);
            array->push_back(temp[1]);
            array->push_back(temp[2]);
        }
    }
    // convert image vector to prep array to copy to GPU
    host_image = array->data();
    // Number of work items in each local work group
    localSize = 64;
    // Number of total work items - localSize must be devisor
    globalSize = ceil(array->size()/(float)localSize)*localSize;
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    // Read kernel
    programHandle = fopen("opencl_kernels.cl", "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);
    // read kernel source into buffer
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**) &programBuffer, &programSize, NULL);
    free(programBuffer);
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "gaussian_blur", &err);
    // Create the input and output arrays in device memory for our calculation
    dev_image = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * array->size(), NULL, NULL);
    dev_result = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * array->size(), NULL, NULL);
    dev_conv = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * kernelSize * kernelSize, NULL, NULL);
    // Copy data into GPU buffers
    err = clEnqueueWriteBuffer(queue, dev_image, CL_TRUE, 0, sizeof(uchar) * array->size(), host_image, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, dev_conv, CL_TRUE, 0, sizeof(float) * kernelSize * kernelSize, conv, 0, NULL, NULL);
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_result);
    err |= clSetKernelArg(kernel, 2, sizeof(uint64), &length);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &kernelSize);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_conv);
    err |= clSetKernelArg(kernel, 5, sizeof(uint64), &frame.rows);
    err |= clSetKernelArg(kernel, 5, sizeof(uint64), &frame.cols);
    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
    // Get results from program
    clEnqueueReadBuffer(queue, dev_result, CL_TRUE, 0, sizeof(uchar) * array->size(), host_image, 0, NULL, NULL );
    // Set threads to maximum available to speed up copying back
    omp_set_num_threads(std::thread::hardware_concurrency());
    //#pragma omp parallel for
    for(int y = 0; y < finalResult.rows; y++) {
        for (int x = 0; x < finalResult.cols; x++) {
            uint64 temp_id = 3 * (y*finalResult.cols + x); // now have 2D to 1D conversion
            cv::Vec3b temp = cv::Vec3b(host_image[temp_id], host_image[temp_id + 1], host_image[temp_id +2]);
            finalResult.at<cv::Vec3b>(y, x) = temp;
        }
    }

    clReleaseMemObject(dev_image);
    clReleaseMemObject(dev_result);
    clReleaseMemObject(dev_conv);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    delete(array);
    delete(conv);

    return finalResult;
}


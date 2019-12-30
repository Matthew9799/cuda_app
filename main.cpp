#include <stdio.h>
#include <QApplication>
#include <QPushButton>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_kernels.hpp"
#include "opencl_kernels.hpp"

using namespace cv;

int main(int argc, const char **argv) {
    Mat frame;
    cudaKernel proc;

    frame = imread("matt.JPG");

    Mat result = proc.gaussian_blur(frame,3,1.0);

    imshow("result", result);
    //Launch GUI to select image

    //assume we have image


    return 0;
}
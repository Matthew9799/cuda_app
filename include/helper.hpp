//
// Created by Matt Lewis on 2019-12-31.
//

#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include "cmath"

#define M_PI 3.14159265358979323846

class helper {
public:
    static void gaussian_convolution(float *arr, int length, float sigma);
};

#endif //CUDA_HELPER_H

//
// Created by Matt Lewis on 2019-12-31.
//

#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include "cmath"

#define M_PI 3.14159265358979323846

class helper {
public:
    static float * gaussian_convolution(int length, float sigma, int flag);
};

#endif //CUDA_HELPER_H

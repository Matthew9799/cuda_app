//
// Created by Matt Lewis on 2019-12-31.
//

#include "../include/helper.hpp"

/*
 * Generate convolution matrix based on desired size and sigma values, flag denotes if one or two d array
 */

float * helper::gaussian_convolution( int length, float sigma, int flag) {
    double r, s = 2.0 * sigma * sigma;
    double sum = 0.0; int index = 0, low = -length/2, high = length/2;
    float * arr = (float*)calloc(length*flag, sizeof(float));
    // generating 5x5 kernel
    for (int x = low; x <= high; x++) {
        for (int y = low; y <= high; y++) {
            r = sqrt(x * x + y * y);
            arr[index] += ((exp(-(r * r) / s)) / (M_PI * s));
        }
        sum += arr[index++];
    }
    // normalising the Kernel
    for (int i = 0; i < length*flag; i++)
        arr[i] /= sum;

    return arr;
}
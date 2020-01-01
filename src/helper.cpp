//
// Created by Matt Lewis on 2019-12-31.
//

#include "../include/helper.hpp"

/*
 * Generate convolution matrix based on desired size and sigma values
 */

void helper::gaussian_convolution(float *arr, int length, float sigma) {
    double r, s = 2.0 * sigma * sigma;
    double sum = 0.0; int index = 0, low = -length/2, high = length/2;
    // generating 5x5 kernel
    for (int x = low; x <= high; x++) {
        for (int y = low; y <= high; y++) {
            r = sqrt(x * x + y * y);
            arr[index] = ((exp(-(r * r) / s)) / (M_PI * s));
            sum += arr[index++];
        }
    }
    // normalising the Kernel
    for (int i = 0; i < length*length; i++)
        arr[i] /= sum;
}
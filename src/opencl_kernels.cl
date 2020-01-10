__kernel void gaussian_blur(__global uchar *image, __global uchar *returnImage, __constant const uint64 length, __constant int kernelSize,
                                 __constant float * conv, __constant uint64 rows, __constant uint64 cols) {

    int global_id = get_global_id(0); // have pixel working on
    double b = 0.0, g = 0.0, r = 0.0;


    if(global_id < length) { // we have not exceeded the range of our array
        for (int y1 = -kernelSize / 2; y1 <= kernelSize / 2; y1++) { // loop through y val of conv matrix
            for (int x1 = -kernelSize / 2; x1 <= kernelSize / 2; x1++) { // loop through x val of conv matrix
                uint64 x = (global_id%cols + x1), y = (global_id/rows + y1); // have master x and y coord
                uint64 temp_id = 3 * (y*cols + x); // now have 2D to 1D conversion
                if (y + y1 >= 0 && y + y1 < rows) { // check to see if out of bounds of rows
                    if (x + x1 >= 0 && x + x1 < cols) {
                        b += image[temp_id] * conv[(kernelSize / 2 + y1) * kernelSize + (kernelSize / 2 + x1)];     // B
                        g += image[temp_id + 1] * conv[(kernelSize / 2 + y1) * kernelSize + (kernelSize / 2 + x1)]; // G
                        r += image[temp_id + 2] * conv[(kernelSize / 2 + y1) * kernelSize + (kernelSize / 2 + x1)]; // R
                    }
                }
            }
        }
        returnImage[global_id*3] = b;
        returnImage[global_id*3 + 1] = g;
        returnImage[global_id*3 + 2] = r;
    }
}
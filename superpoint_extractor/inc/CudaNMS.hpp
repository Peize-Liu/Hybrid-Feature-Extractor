#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
void cudaNMS(unsigned char* image, int width, int height, 
    int neighborhood_size, unsigned char feature_threshold);
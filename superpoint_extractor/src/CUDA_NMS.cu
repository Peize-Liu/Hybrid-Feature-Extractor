#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void nms_kernel(float* dev_image, int width, int height, int neighborhood_size, unsigned char feature_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // Boundary check

    unsigned char value = dev_image[y * width + x];
    if (value <= feature_threshold) {
        dev_image[y * width + x] = 0; // Suppress if below the threshold
        return;
    }
    // Check if there exists a higher value within the neighborhood
    int half_size = neighborhood_size / 2;
    for (int dy = -half_size; dy <= half_size; ++dy) {
        for (int dx = -half_size; dx <= half_size; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (dev_image[ny * width + nx] > value) {
                    dev_image[y * width + x] = 0; // Suppress this point because a higher value exists
                    return;
                }
            }
        }
    }
}

void cudaNMS(unsigned char* image, int width, int height, int neighborhood_size, unsigned char feature_threshold) {
    unsigned char* dev_image;
    size_t image_size = sizeof(unsigned char) * width * height;

    // Allocate and transfer to device memory
    auto start = std::chrono::high_resolution_clock::now();
    cudaMallocManaged(&dev_image, image_size);
    cudaMemcpy(dev_image, image, image_size, cudaMemcpyHostToDevice);

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout << "NMS memory transfer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms" << std::endl;

    // Configure CUDA kernel launch parameters
    start = std::chrono::high_resolution_clock::now();
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout << "NMS kernel configuration time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms" << std::endl;

    // Launch the kernel
    start = std::chrono::high_resolution_clock::now();
    nms_kernel<<<numBlocks, threadsPerBlock>>>(dev_image, width, height, neighborhood_size, feature_threshold);
    cudaDeviceSynchronize(); // Wait for the CUDA kernel to finish
    elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout << "NMS kernel time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms" << std::endl;

    // Copy back to host memory
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(image, dev_image, image_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_image); // Release device memory
    elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout << "NMS memory transfer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms" << std::endl;
}
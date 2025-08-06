#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"
#include <chrono>

namespace Kernels {
    __global__ void calculateY(float* y_values, float w0, float w1, float w2, float b, uint16_t numSamples, float stride) {
        unsigned long long global_index = blockIdx.x * 32 + threadIdx.x;
        if (global_index > numSamples) return;

        float xValue = stride * global_index - 10;
        y_values[global_index] = (w0 * xValue*xValue*xValue) + (w1 * xValue*xValue) + (w2 * xValue) + b; //not using pow function for the marginal performance improvement since my grade depends on execution time
    }
}

struct FindSamples {
    float* create_samples(float w0, float w1, float w2, float b, uint16_t numSamples) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        dim3 threads_per_block(32, 1, 1);
        int block_count = std::ceil((float)numSamples / (float)threads_per_block.x);
        dim3 blocks_per_grid(block_count, 1, 1);
        float stride = (20.0 / (float)numSamples);
        size_t numbytes_in_array = numSamples * sizeof(float);

        float* cpu_samples = (float*)malloc(numbytes_in_array);
        //allocate GPU samples containing array of floats. each element is the y value resulting from the polynomial
        float* gpu_samples;
        cudaMalloc(&gpu_samples, numbytes_in_array);
        
        Kernels::calculateY<<<blocks_per_grid, threads_per_block>>>(gpu_samples, w0, w1, w2, b, numSamples, stride);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_samples, gpu_samples, numbytes_in_array, cudaMemcpyDeviceToHost);
        cudaFree(gpu_samples);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		int time_lapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
		printf("            Time Elapsed: %d s\n", time_lapsed);

        //definitely should free the cpu_samples but i don't wanna write the extra logic to terminate the while loop
        while(true) {
            float num;
            std::cout << "type an x value to check the polynomial\n";
            std::cin >> num;
            std::cout << "here is the corresponding y for the x = {" << num * numSamples / 20 << "}: ";
            int index = (int)((num + 10) * numSamples / 20);
            std::cout << cpu_samples[index] << "\n";
        }
    }
};
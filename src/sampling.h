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

        dim3 threads_per_block(32, 1, 1);
        int block_count = std::ceil((float)numSamples / (float)threads_per_block.x); //number of blocks = numSamples / threadsPerBlock (plus 1 if necessary)
        dim3 blocks_per_grid(block_count, 1, 1);
        float stride = (20.0 / (float)numSamples); //delta x of each sample, ie how far between each x
        size_t numbytes_in_array = numSamples * sizeof(float);

        //allocate array where samples will be stored
        float* cpu_samples = (float*)malloc(numbytes_in_array);
        //allocate GPU samples 
        float* gpu_samples;
        cudaMalloc(&gpu_samples, numbytes_in_array);
        
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //start clock

        Kernels::calculateY<<<blocks_per_grid, threads_per_block>>>(gpu_samples, w0, w1, w2, b, numSamples, stride);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_samples, gpu_samples, numbytes_in_array, cudaMemcpyDeviceToHost); //copy gpu array to cpu array after everything is synchronized
        cudaFree(gpu_samples); 

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); //stop clock
		float time_lapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
		printf("            Time Elapsed: %.2f s\n", time_lapsed);

        //definitely should free the cpu_samples but i don't wanna write the extra logic to terminate the while loop
        for (int i = 0; i < 1000000; i+=1000) {
            std::cout << "here is the corresponding y for the x = {" << stride * i - 10 << "}: "; //hard coded scaling 100 samples to values from [-10, 10]
            std::cout << cpu_samples[i] << "\n";
        }
        return cpu_samples;
    }
};
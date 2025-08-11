#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"
#include <chrono>

namespace Kernels {
    __global__ void calculateY(double* y_values, double w0, double w1, double w2, double b, uint32_t numSamples, double stride, uint16_t job_size) {
        unsigned long long global_index = blockIdx.x * 32 + threadIdx.x;
        if (global_index >= numSamples) return; 

        for (int i = 0; i < job_size; i++) {
            int arr_index = global_index * job_size + i;
            if (arr_index >= numSamples) return;
            double xValue = stride * arr_index - 10; 
            y_values[arr_index] = (w0 * xValue*xValue*xValue) + (w1 * xValue*xValue) + (w2 * xValue) + b; //not using pow function for the marginal performance improvement since my grade depends on execution time
        }
    }
}

struct FindSamples {
    /** w0 through b are coefficients of the polynomial. numSamples is how many x values to calculate from [-10, 10]. job_size is how many x values each thread is responsible for */
    double* create_samples(double w0, double w1, double w2, double b, uint32_t numSamples, uint16_t job_size) {
        std::cout << "hi";
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //start clock

        dim3 threads_per_block(32, 1, 1);
        int block_count = std::ceil((double)numSamples / (double)job_size / (double)threads_per_block.x); //number of blocks = numSamples / threadsPerBlock (plus 1 if necessary)
        std::cout << "block count: " << block_count << "\n";
        dim3 blocks_per_grid(block_count, 1, 1);
        double stride = (20.0 / (double)numSamples); //delta x of each sample, ie how far between each x
        std::cout << "stride: " << stride << "\n";
        size_t numbytes_in_array = numSamples * sizeof(double);

        //allocate array where samples will be stored
        double* cpu_samples = (double*)malloc(numbytes_in_array);
        //allocate GPU samples 
        double* gpu_samples;
        cudaMalloc(&gpu_samples, numbytes_in_array);

        Kernels::calculateY<<<blocks_per_grid, threads_per_block>>>(gpu_samples, w0, w1, w2, b, numSamples, stride, job_size);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_samples, gpu_samples, numbytes_in_array, cudaMemcpyDeviceToHost); //copy gpu array to cpu array after everything is synchronized
        cudaFree(gpu_samples); 

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); //stop clock
		double time_lapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		printf("            Time Elapsed: %.2f s\n", time_lapsed * pow(10, -9));

        //definitely should free the cpu_samples but i don't wanna write the extra logic to terminate the while loop
        uint32_t increment = numSamples / 100;
        if (increment == 0) increment = 1;
        for (int i = 0; i < numSamples; i+= increment) {
            std::cout << "here is the corresponding y for the x = {" << stride * i - 10 << "}: "; //hard coded scaling 100 samples to values from [-10, 10]
            std::cout << cpu_samples[i] << "\n";
        }
        std::cout << cpu_samples[565587000] << "\n";
        return cpu_samples;
    }
};
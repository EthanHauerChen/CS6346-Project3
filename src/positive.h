#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"
#include <chrono>

namespace Kernels {
    __global__ void detectPositive(double* samples, bool* pos_or_neg, uint32_t numSamples, uint16_t job_size) {
        unsigned long long global_index = blockIdx.x * 32 + threadIdx.x;
        if (global_index >= numSamples) return; 

        for (int i = 0; i < job_size; i++) {
            int arr_index = global_index * job_size + i;
            if (arr_index >= numSamples) return;
            pos_or_neg[arr_index] = samples[arr_index] > 0;
        }
    }
}

struct FindPositives {
    /** w0 through b are coefficients of the polynomial. numSamples is how many x values to calculate from [-10, 10]. job_size is how many x values each thread is responsible for */
    uint32_t detectPositive(double* samples, uint32_t numSamples, uint16_t job_size) {
        std::cout << "start positive \n";
        uint32_t num_positive = 0;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //start clock

        dim3 threads_per_block(32, 1, 1);
        int block_count = std::ceil((double)numSamples / (double)job_size / (double)threads_per_block.x); //number of blocks = numSamples / threadsPerBlock (plus 1 if necessary)
        dim3 blocks_per_grid(block_count, 1, 1);
        double stride = (20.0 / (double)(numSamples-1)); //delta x of each sample, ie how far between each x
        size_t numbytes_in_polynomial = numSamples * sizeof(double);

        size_t numbytes_in_bool = numSamples * sizeof(bool);
        //allocate bool array containing true if the polynomial at that x value is positive, false otherwise
        bool* cpu_pos_or_neg = (bool*)malloc(numbytes_in_bool);
        //allocate GPU bool array containing true if the polynomial at that x value is positive, false otherwise
        bool* gpu_pos_or_neg;
        cudaMalloc(&gpu_pos_or_neg, (size_t)numbytes_in_bool);
        //allocate GPU samples 
        double* gpu_samples;
        cudaMalloc(&gpu_samples, numbytes_in_polynomial);
        cudaMemcpy(gpu_samples, samples, numbytes_in_polynomial, cudaMemcpyHostToDevice);
        
        Kernels::detectPositive<<<blocks_per_grid, threads_per_block>>>(gpu_samples, gpu_pos_or_neg, numSamples, job_size);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_pos_or_neg, gpu_pos_or_neg, numbytes_in_bool, cudaMemcpyDeviceToHost); //copy gpu array to cpu array after everything is synchronized
        cudaFree(gpu_samples); 
        cudaFree(gpu_pos_or_neg);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); //stop clock

		double time_lapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		printf("            Counting Positive Samples Time Elapsed: %.2f s\n", time_lapsed * pow(10, -9));

        //definitely should free the cpu_samples but i don't wanna write the extra logic to terminate the while loop
        // uint32_t increment = numSamples / 100;
        // if (increment < 1) increment = 1;
        // for (int i = 0; i < numSamples; i+= increment) {
        //     std::cout << "the polynomial at x = {" << stride * i - 10 << "} is positive: "; //hard coded scaling 100 samples to values from [-10, 10]
        //     printf("%s\n", cpu_pos_or_neg[i] ? "true" : "false");
        // }

        // for (uint32_t i = 0; i < numSamples; i++) 
        //     if (cpu_pos_or_neg[i]) num_positive++;
        return num_positive;
    }
};
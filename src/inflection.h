#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"
#include <chrono>

namespace Kernels {
    __global__ void calc_first_derivative(float* derivatives, float w0, float w1, float w2, uint32_t numSamples, float stride, uint16_t job_size) {
        unsigned long long global_index = blockIdx.x * 32 + threadIdx.x;
        if (global_index >= numSamples) return; 

        for (int i = 0; i < job_size; i++) {
            int arr_index = global_index * job_size + i;
            if (arr_index >= numSamples) return;
            float xValue = stride * arr_index - 10; 
            //calulate second derivate using power rule
            derivatives[arr_index] = (3*w0 * xValue*xValue) + (2*w1 * xValue) + w2;
        }
        
    }
    
    __global__ void calc_second_derivative(float* derivatives, float w0, float w1, uint32_t numSamples, float stride, uint16_t job_size) {
        unsigned long long global_index = blockIdx.x * 32 + threadIdx.x;
        if (global_index >= numSamples) return; 

        for (int i = 0; i < job_size; i++) {
            int arr_index = global_index * job_size + i;
            if (arr_index >= numSamples) return;
            float xValue = stride * arr_index - 10; 
            //calulate second derivate using power rule
            derivatives[arr_index] = (6*w0 * xValue) + 2*w1;
        }
        
    }

    __global__ void search_inflection_points(float* derivatives, bool* is_inflection_point, float numSamples, float stride, uint16_t job_size) {
        unsigned long long global_index = blockIdx.x * 32 + threadIdx.x;
        if (global_index >= numSamples) return; 

        // if (derivatives[global_index-1] * derivatives[global_index+1] < 0) { //if result is negative, that means the 2nd derivative switches sign from either side
        //     is_inflection_point[global_index] = true;
        //     return;
        // }
        // //else
        // is_inflection_point[global_index] = false;
        // return;

        for (int i = 0; i < job_size; i++) {
            int arr_index = global_index * job_size + i;
            if (arr_index > numSamples-2) return;
            if (arr_index == 0) continue;
            
            if ((derivatives[arr_index-1] > 0 && derivatives[arr_index+1] < 0) || (derivatives[arr_index-1] < 0 && derivatives[arr_index+1] > 0))
                is_inflection_point[arr_index] = true;
            else
                is_inflection_point[arr_index] = false;
        }
        return;
    }
}

struct FindInflections {
    /** w0 through b are coefficients of the polynomial. numSamples is how many x values to calculate from [-10, 10]. job_size is how many x values each thread is responsible for */
    bool* detect_inflection_points(float w0, float w1, float w2, uint32_t numSamples, uint16_t job_size) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //start clock

        dim3 threads_per_block(32, 1, 1);
        int block_count = std::ceil((float)numSamples / (float)job_size / (float)threads_per_block.x); //number of blocks = numSamples / threadsPerBlock (plus 1 if necessary)
        dim3 blocks_per_grid(block_count, 1, 1);
        float stride = (20.0 / (float)numSamples); //delta x of each sample, ie how far between each x
        size_t numbytes_in_polynomial = numSamples * sizeof(float);
        size_t numbytes_in_bool = numSamples * sizeof(bool);
        
        //allocate bool array containing true if inflection point
        bool* cpu_inflection = (bool*)malloc(numbytes_in_bool);
        bool* gpu_inflection;
        cudaMalloc(&gpu_inflection, (size_t)numbytes_in_bool);

        //allocate derivatives
        float* cpu_derivatives = (float*)malloc(numbytes_in_polynomial);
        float* gpu_derivatives;
        cudaMalloc(&gpu_derivatives, numbytes_in_polynomial);
        
        Kernels::calc_second_derivative<<<blocks_per_grid, threads_per_block>>>(gpu_derivatives, w0, w1, numSamples, stride, job_size);
        cudaMemcpy(cpu_derivatives, gpu_derivatives, numbytes_in_polynomial, cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_derivatives, cpu_derivatives, numbytes_in_polynomial, cudaMemcpyHostToDevice);
        Kernels::search_inflection_points<<<blocks_per_grid, threads_per_block>>>(gpu_derivatives, gpu_inflection, numSamples, stride, job_size);
        cudaMemcpy(cpu_inflection, gpu_inflection, numbytes_in_bool, cudaMemcpyDeviceToHost);
        cudaFree(gpu_derivatives);
        cudaFree(gpu_inflection);
        //free(cpu_derivatives);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); //stop clock

		float time_lapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		printf("            Time Elapsed: %.2f s\n", time_lapsed * pow(10, -9));

        //definitely should free the cpu_samples but i don't wanna write the extra logic to terminate the while loop
        for (int i = 0; i < numSamples; i+= (numSamples/100)) {
            std::cout << "the polynomial at x = {" << stride * i - 10 << "} has inflection point: "; //hard coded scaling 100 samples to values from [-10, 10]
            printf("%s\n", cpu_inflection[i] ? "true" : "false");
        }
        std::cout << "the polynomial at x = {.8333333} has inflection point: "; //hard coded scaling 100 samples to values from [-10, 10]
        printf("%s\n", cpu_inflection[458333332] ? "true" : "false");
        std::cout << "the polynomial at x = {.8333333} has inflection point: "; //hard coded scaling 100 samples to values from [-10, 10]
        printf("%s\n", cpu_inflection[458333333] ? "true" : "false");
        std::cout << "the polynomial at x = {.8333333} has inflection point: "; //hard coded scaling 100 samples to values from [-10, 10]
        printf("%s\n", cpu_inflection[458333334]) ? "true" : "false");
        return cpu_inflection;
    }
};
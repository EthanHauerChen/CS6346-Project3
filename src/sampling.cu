#include <chrono>

namespace Kernels {
    __global__ void calculateY(float* y_values, float w0, float w1, float w2, float b, uint16_t numSamples, float stride) {
        unsigned long long global_index = blockIdx * 32 + threadIdx;
        if (global_index > numSamples) return;

        float xValue = stride * global_index - 10;
        y_values[global_index] = (w0 * xValue*xValue*xValue) + (w1 * xValue*xValue) + (w2 * xValue) + b; //not using pow function for the marginal performance improvement since my grade depends on execution time
    }
}

struct FindSamples {
    float* createSamples(float w0, float w1, float w2, float b, uint16_t numSamples) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        dim3 threads_per_block(32, 1, 1);
        int block_count = std::ceil((float)numSamples / (float)threads_per_block.x);
        dim3 blocks_per_grid(block_count, 1, 1);
        float stride = (20f / (float)numSamples);

        float* cpu_samples;
        //allocate GPU samples containing array of floats. each element is the y value resulting from the polynomial
        float* gpu_samples;
        cudaMalloc(&gpu_samples, (size_t) (numSamples * sizeof(float)));
        
        Kernels::calculateY<<<blocks_per_grid, threads_per_block>>>(gpu_samples, w0, w1, w2, b, stride);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		int time_lapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
		printf("            Time Elapsed: %d s\n", time_lapsed);
    }
}
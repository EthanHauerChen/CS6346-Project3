#include <iostream>

#include "sampling.h"
#include "positive.h"

struct polynomial {
        short w0,
        short w1,
        short w2,
        short b
};

int main() {
    polynomial p{1, -1, 0.5, -.5, 1};
    uint32_t numSamples = 1000000000;
    uint16_t num_jobs = 1;
    FindSamples sampling_object{};
    FindPositives positive_object{};

    float* samples = sampling_object.create_samples(p->w0, p->w1, p->w2, p->b, numSamples, num_jobs);
    positive_object.detect_positive(samples, numSamples, num_jobs);

    return 0;
}
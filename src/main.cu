#include <iostream>

#include "sampling.h"
#include "positive.h"
#include "inflection.h"

struct polynomial {
        float w0;
        float w1;
        float w2;
        float b;
};

int main() {
    polynomial p{.2, .5, -1, 0};
    uint32_t numSamples = 1000000000;
    uint16_t num_jobs = 1;
    FindSamples sampling_object{};
    FindInflections inflection_object{};
    FindPositives positive_object{};

    float* samples = sampling_object.create_samples(p.w0, p.w1, p.w2, p.b, numSamples, num_jobs);
    inflection_object.detect_inflection_points(p.w0, p.w1, numSamples, num_jobs);
    positive_object.detectPositive(samples, numSamples, num_jobs);

    return 0;
}
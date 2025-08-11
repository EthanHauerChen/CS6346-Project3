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

/* return corresponding y value of the polynomial given an x*/
double get_y(double* samples, double x_value, uint32_t numSamples) {
    uint32_t index = (x_value + 10) / (20 / numSamples);
    return samples[index];
}

int main() {
    polynomial p{.2, .5, -1, 0};
    uint32_t numSamples = 10000;
    uint16_t num_jobs = 1;
    FindSamples sampling_object{};
    FindInflections inflection_object{};
    FindPositives positive_object{};

    double* samples = sampling_object.create_samples(p.w0, p.w1, p.w2, p.b, numSamples, num_jobs);
    bool* inflection_points = inflection_object.detect_inflection_points(p.w0, p.w1, p.w2, numSamples, num_jobs);
    uint32_t num_positive = positive_object.detectPositive(samples, numSamples, num_jobs);

    return 0;
}
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
uint32_t get_index(double x_value, uint32_t numSamples) {
    uint32_t index = (x_value + 10) / (20.0 / numSamples);
    return index;
}

int main(int argc, char* argv[]) {
    polynomial p{.2, .5, -1, 0};
    uint32_t numSamples = 1000000000;
    uint16_t num_jobs = 1;
    FindSamples sampling_object{};
    FindInflections inflection_object{};
    FindPositives positive_object{};

    double* samples = sampling_object.create_samples(p.w0, p.w1, p.w2, p.b, numSamples, num_jobs);
    bool* inflection_points = inflection_object.detect_inflection_points(p.w0, p.w1, p.w2, numSamples, num_jobs);
    uint32_t num_positive = positive_object.detectPositive(samples, numSamples, num_jobs);

    int argi = 0;
    while (argi < argc) {
        double x_value = atof(argv[argi]);
        uint32_t index = get_index(x_value, numSamples);
        std::cout << "index: " << index << "\n";
        std::cout << "polynomial at x = " << x_value << ": (" << x_value << ", " << samples[get_index(x_value, numSamples)] << ")\n";
        argi++;
    }
    std::cout << "polynomial at (-10 <= x <= 10), in increments of 0.2: (x, y)\n";
    for (uint i = 0; i < numSamples; i += (numSamples / 100)) {
        std::cout << "(" << (20.0 / numSamples) * i - 10 << ", " << samples[i] << ")\n";
    }



    return 0;
}
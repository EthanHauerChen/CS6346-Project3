#include <iostream>
#include <cmath>

#include "sampling.h"
#include "positive.h"
#include "inflection.h"

struct polynomial {
        float w0;
        float w1;
        float w2;
        float b;
};

/* return corresponding index of the array given an x*/
uint32_t get_index(double x_value, uint32_t numSamples) {
    uint32_t index = (x_value + 10) / (20.0 / (numSamples-1));
    return index;
}

/* return corresponding x value of the polynomial given an array index*/
double get_x(uint32_t index, uint32_t numSamples) {
    double x = (index * (20.0 / (numSamples-1))) - 10;
    return x;
}

void render_graph(double* samples, uint32_t numSamples) {
    uint32_t scaled_x = (numSamples - 1) / 20;
    if (scaled_x == 0) scaled_x = 1;

    char graph [11][21];
    for (int r = 0; r < 11; r++) {
        for (int c = 0; c < 21; c++) graph[r][c] = '.';
    }
    for (int c = 0; c < 21; c++) {
        uint32_t arr_index = scaled_x * c;
        if (numSamples < 21) arr_index /= numSamples;
        if (samples[arr_index] > 5 || samples[arr_index] < -5) continue;
        else {
            std::cout << "arr_index = " << arr_index << " y=" << samples[arr_index] << "\n";
            graph[4 - (uint32_t)std::round(samples[arr_index])][c] = '#';
        }
    }

    for (int r = 0; r < 11; r++) {
        for (int c = 0; c < 21; c++) std::cout << graph[r][c];
        std::cout << '\n';
    }
}

int main(int argc, char* argv[]) {
    polynomial p{.2, .5, -1, -.5};
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
        std::cout << "polynomial at x = " << x_value << ": (" << x_value << ", " << samples[get_index(x_value, numSamples)] << ")\n";
        argi++;
    }
    std::cout << "\n\npolynomial at (-10 <= x <= 10), in increments of 0.2 (unless there's less than 100 samples): (x, y)\n";
    uint32_t increment = numSamples / 100;
    if (increment < 1) increment = 1;
    for (uint32_t i = 0; i < numSamples; i += increment) {
        std::cout << "(" << (20.0 / (numSamples-1)) * i - 10 << ", " << samples[i] << ")\n";
    }

    std::cout << "\n\ninflection points: \n";
    for (uint32_t i = 0; i < numSamples; i++) 
        if (inflection_points[i]) std::cout << "inflection point at (" << get_x(i, numSamples) << ", " << samples[i] << ")\n";

    std::cout << "\n\n";
    std::cout << num_positive << " positive samples\n\n";

    render_graph(samples, numSamples);

    return 0;
}
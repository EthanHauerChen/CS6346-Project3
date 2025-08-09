#include <iostream>

#include "sampling.h"

int main() {
    FindSamples sampling_object{};
    sampling_object.create_samples(1, 2, 3, 4, 1000, 1);

    return 0;
}
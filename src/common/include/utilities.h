// src/common/include/utilities.h

#pragma once
#include <iostream>
#include <cuda_runtime.h>

// This macro wraps a CUDA API call and checks its return value.
// If the call results in an error, it prints a descriptive message
// with the file and line number and then exits the program.
#define CUDA_CHECK_ERROR(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
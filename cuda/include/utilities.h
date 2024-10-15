// include/utilities.h

#ifndef UTILITIES_H
#define UTILITIES_H

#include <cuda_runtime.h>
#include <iostream>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK_ERROR(call)                                    \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__    \
                      << std::endl;                               \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#endif // UTILITIES_H
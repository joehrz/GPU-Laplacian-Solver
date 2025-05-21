// cuda/src/solution_export.cpp

#include "solution_export.h" 
#include "utilities.h"        // for CUDA_CHECK_ERROR
#include <vector>
#include <iostream>          // For std::cerr (if error opening file in the host func) & std::cout
#include <cuda_runtime.h>


void exportDeviceSolutionToCSV(const double* d_ptr, 
                         int width,
                         int height,
                         const std::string& filename,
                         const std::string& solverName)
{
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    std::vector<double> hostData(width * height);
    CUDA_CHECK_ERROR(cudaMemcpy(hostData.data(),
                                d_ptr,
                                width * height * sizeof(double),
                                cudaMemcpyDeviceToHost));

    exportHostDataToCSV(hostData.data(), width, height, filename, solverName);

}

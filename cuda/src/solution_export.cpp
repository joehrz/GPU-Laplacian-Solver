// src/solution_export.cpp

#include "solution_export.h"
#include "utilities.h"        // for CUDA_CHECK_ERROR, if needed
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

void exportSolutionToCSV(const double* d_ptr,
                         int width,
                         int height,
                         const std::string& filename,
                         const std::string& solverName)
{
    // 1) Synchronize the device, ensuring all kernels done
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 2) Copy device data to a host vector
    std::vector<double> hostData(width * height);
    CUDA_CHECK_ERROR(cudaMemcpy(hostData.data(),
                                d_ptr,
                                width * height * sizeof(double),
                                cudaMemcpyDeviceToHost));

    // 3) Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[" << solverName << "] Error: Cannot open file "
                  << filename << " for writing.\n";
        return;
    }

    file << std::fixed << std::setprecision(6);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int idx = i + j * width;
            file << hostData[idx];
            if (i < width - 1) file << ",";
        }
        file << "\n";
    }
    file.close();

    std::cout << "[" << solverName << "] Solution exported to "
              << filename << ".\n";
}

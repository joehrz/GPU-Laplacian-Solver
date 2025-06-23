// common/src/solution_export.cpp

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef SOLVER_ENABLE_CUDA          
#include <cuda_runtime.h>
#endif

/* ---------------------------------------------------------------- *\
   Host-buffer export  (always available)
\* ---------------------------------------------------------------- */
void exportHostDataToCSV(const float* h_data,
                         int           width,
                         int           height,
                         const std::string& filename,
                         const std::string& solverName)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << '[' << solverName << "] Error: cannot open "
                  << filename << " for writing.\n";
        return;
    }

    file << std::fixed << std::setprecision(6);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            file << h_data[j * width + i];
            if (i + 1 != width) file << ',';
        }
        file << '\n';
    }
    std::cout << '[' << solverName << "] Solution exported to "
              << filename << ".\n";
}

/* ---------------------------------------------------------------- *\
   Device-buffer export
   – real implementation when CUDA is enabled
   – graceful stub otherwise
\* ---------------------------------------------------------------- */
#ifdef SOLVER_ENABLE_CUDA
void exportDeviceSolutionToCSV(const float* d_ptr,
                               int           width,
                               int           height,
                               const std::string& filename,
                               const std::string& solverName)
{
    std::vector<float> host(width * height);
    cudaError_t err = cudaMemcpy(host.data(), d_ptr,
                                 width * height * sizeof(float),
                                 cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        std::cerr << '[' << solverName << "] cudaMemcpy failed: "
                  << cudaGetErrorString(err) << '\n';
        return;
    }
    exportHostDataToCSV(host.data(), width, height, filename, solverName);
}
#else
/* ---- CPU-only build: provide a stub so the linker is happy ---- */
void exportDeviceSolutionToCSV(const float*,
                               int, int,
                               const std::string&,
                               const std::string& solverName)
{
    std::cerr << '[' << solverName << "] CUDA support not compiled in; "
              << "cannot export device buffer.\n";
}
#endif
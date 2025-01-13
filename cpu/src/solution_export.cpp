#include "solution_export.h"   // Adjust include path if needed

#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>

// CPU-only version (no CUDA calls)
void exportSolutionToCSV(const double* dataPtr,
                         int width,
                         int height,
                         const std::string& filename,
                         const std::string& solverName)
{
    // 1) Check input
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[" << solverName << "] Error: Cannot open file "
                  << filename << " for writing.\n";
        return;
    }

    // 3) Write to CSV
    file << std::fixed << std::setprecision(6);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int idx = i + j * width;
            file << dataPtr[idx];  // or hostData[idx] if you made a copy
            if (i < width - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();

    std::cout << "[" << solverName << "] Solution exported to "
              << filename << ".\n";
}
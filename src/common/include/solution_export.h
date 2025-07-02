// src/common/include/solution_export.h

#pragma once
#include <string>

void exportHostDataToCSV(const float* h_data, int w, int h,
                         const std::string& file,
                         const std::string& solver = "GenericSolver");

void exportDeviceSolutionToCSV(const float* d_ptr, int w, int h,
                               const std::string& file,
                               const std::string& solver = "CUDASolver");
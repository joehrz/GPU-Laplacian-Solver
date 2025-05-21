// include/solution_export.h
#ifndef SOLUTION_EXPORT_H
#define SOLUTION_EXPORT_H

#include <string>

/* ---------- Host-only helper ---------- */
void exportHostDataToCSV(const double* h_data,
                         int           width,
                         int           height,
                         const std::string& filename,
                         const std::string& solverName = "GenericSolver");

/* ---------- Device-buffer helper (prototype only) ---------- */
void exportDeviceSolutionToCSV(const double* d_ptr,
                               int           width,
                               int           height,
                               const std::string& filename,
                               const std::string& solverName = "CUDASolver");

#endif // SOLUTION_EXPORT_H
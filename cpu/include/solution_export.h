// include/solution_export.h

#ifndef SOLUTION_EXPORT_H
#define SOLUTION_EXPORT_H

#include <string>

/**
 * @brief Exports the solver’s final device array to a CSV file.
 *
 * @param d_ptr    Pointer to the device (GPU) data (width*height doubles).
 * @param width    Grid width.
 * @param height   Grid height.
 * @param filename Path to CSV file to create.
 * @param solverName Optional string to identify which solver is exporting.
 */
void exportSolutionToCSV(const double* d_ptr,
                         int width,
                         int height,
                         const std::string& filename,
                         const std::string& solverName = "GenericSolver");

#endif // SOLUTION_EXPORT_H

#ifndef SOR_METHODS_H
#define SOR_METHODS_H

#include <vector>
#include <string>

// Constants (declare them as extern)
extern const int M;
extern const int N;
extern const double OMEGA;
extern const int MAX_ITER;
extern const double TOL;

// Function declarations
double analyticalSolution(int x, int y, int M, int N);
void validateSolution(const std::vector<std::vector<double>>& grid);
void initializeGrid(std::vector<std::vector<double>>& grid);
void updateStandardSOR(std::vector<std::vector<double>>& grid);
void updateRedBlackSOR(std::vector<std::vector<double>>& grid);
void exportSolution(const std::vector<std::vector<double>>& grid, const std::string& filename);
double timeSOR(void (*sorMethod)(std::vector<std::vector<double>>&), std::vector<std::vector<double>>& grid);

#endif // SOR_METHODS_H


#ifndef SOR_METHODS_H
#define SOR_METHODS_H

#include <vector>
#include <string>

// Constants
constexpr int M = 100;
constexpr int N = 100;
constexpr int MAX_ITER = 10000;
constexpr double OMEGA = 1.85;
constexpr double TOL = 1e-6;

// Function declarations
double analyticalSolution(int x, int y, int M, int N);
void validateSolution(const std::vector<std::vector<double>>& grid);
void initializeGrid(std::vector<std::vector<double>>& grid);
void updateStandardSOR(std::vector<std::vector<double>>& grid);
void updateRedBlackSOR(std::vector<std::vector<double>>& grid);
double timeSOR(void (*sorMethod)(std::vector<std::vector<double>>&), std::vector<std::vector<double>>& grid);
void exportSolution(const std::vector<std::vector<double>>& grid, const std::string& filename);
void plotSolution(const std::vector<std::vector<double>> &solution, int M, int N, const char *title);


#endif // SOR_METHODS_H
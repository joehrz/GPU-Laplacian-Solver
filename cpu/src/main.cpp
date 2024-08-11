#include <iostream> // Include this to use std::cout
#include "sor_methods.h"

int main() {
    std::vector<std::vector<double>> phi_sor(M, std::vector<double>(N));
    std::vector<std::vector<double>> phi_sor_red_black(M, std::vector<double>(N));

    // Standard SOR
    initializeGrid(phi_sor);
    double timeStandardSOR = timeSOR(updateStandardSOR, phi_sor);
    std::cout << "Standard SOR time: " << timeStandardSOR << " seconds.\n";
    validateSolution(phi_sor);
    exportSolution(phi_sor, "standard_sor.dat");

    // Red-Black SOR
    initializeGrid(phi_sor_red_black);
    double timeRedBlackSOR = timeSOR(updateRedBlackSOR, phi_sor_red_black);
    std::cout << "Red-Black SOR time: " << timeRedBlackSOR << " seconds.\n";
    validateSolution(phi_sor_red_black);
    exportSolution(phi_sor_red_black, "red_black_sor.dat");

    // Analytical Solution
    std::vector<std::vector<double>> analyticalGrid(M, std::vector<double>(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            analyticalGrid[i][j] = analyticalSolution(i, j, M, N);
        }
    }
    exportSolution(analyticalGrid, "analytical.dat");

    // Plot numerical solution
    plotSolution(phi_sor, M, N, "Numerical Solution");

    // Plot analytical solution
    plotSolution(analyticalGrid, M, N, "Analytical Solution");

    return 0;
}

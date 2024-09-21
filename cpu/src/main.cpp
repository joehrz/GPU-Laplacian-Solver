#include "sor_methods.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>  // For std::string

using namespace std;

int main() {
    // Declare and initialize the main grid
    std::vector<std::vector<double>> grid(M, std::vector<double>(N, 0.0));

    // Initialize the grid with boundary conditions
    initializeGrid(grid);

    // Time the Standard SOR method
    std::vector<std::vector<double>> gridStandard = grid;  // Copy grid for standard SOR
    double standardTime = timeSOR(updateStandardSOR, gridStandard);
    cout << "Standard SOR took " << standardTime << " seconds.\n";

    // Time the Red-Black SOR method
    std::vector<std::vector<double>> gridRedBlack = grid;  // Copy grid for red-black SOR
    double redBlackTime = timeSOR(updateRedBlackSOR, gridRedBlack);
    cout << "Red-Black SOR took " << redBlackTime << " seconds.\n";

    // Validate the Red-Black SOR solution
    validateSolution(gridRedBlack);

    // Export the solved Standard SOR solution and plot
    exportSolutionAndPlot(gridStandard, "cpu_standard_sor_solution.txt", "Standard SOR Solution");

    // Export the solved Red-Black SOR solution and plot
    exportSolutionAndPlot(gridRedBlack, "cpu_red_black_sor_solution.txt", "Red and Black SOR Solution");

    return 0;
}


// cpu/src/main.cpp


#include "boundary_conditions.h"      // Centralized
#include "grid_initialization.h"     // Centralized
#include "solver_base.h"             // Centralized

//#include "sor_methods.h"

#include "solver_basic.h"            // CPU-specific
#include "solver_red_black.h"        // CPU-specific


#include <iostream>
#include <vector>
#include <chrono>
#include <string>  // For std::string
#include <filesystem>  // Add this line

using namespace std;



namespace fs = std::filesystem;

std::string getProjectDir(int levels_up = 3) {
    fs::path current_dir = fs::current_path();
    fs::path project_dir = current_dir;

    for(int i = 0; i < levels_up; ++i){
        project_dir = project_dir.parent_path();
        if(project_dir.empty()){
            throw std::runtime_error("Cannot move up any further in the directory structure.");
        }
    }

    return project_dir.string();
}


int main(int argc, char* argv[]){
    // Declare variables outside the try block to ensure they are in scope
    std::string project_dir;
    std::string script_path;
    std::string bc_file;

    try {
        project_dir = getProjectDir(); // Default is 3 levels up
        std::cout << "Project Directory: " << project_dir << "\n";

        // Constructing paths relative to the project directory
        fs::path bc_file_path = fs::path(project_dir) / "boundary_conditions" / "boundary_conditions.json";
        bc_file = bc_file_path.string();
        std::cout << "Boundary Conditions File Path: " << bc_file << "\n";

        fs::path script_path_fs = fs::path(project_dir) / "scripts" / "plot_solution.py";
        script_path = script_path_fs.string();
        std::cout << "Plotting Solutions File Path: " << script_path << "\n";
    }
    catch(const std::exception& e){
        std::cerr << "Exception during project directory setup: " << e.what() << "\n";
        return EXIT_FAILURE;
    }


    try {


        if (argc > 1){
            bc_file = argv[1];
        }

        BoundaryConditions bc = loadBoundaryConditions(bc_file);

        // Default grid dimensions
        const int width = 100;
        const int height = 100;

        // Or better, using std::vector
        std::vector<double> U(width * height, 0.0);
        // Initialize the grid with boundary conditions       
        initializeGrid(U.data(), width, height, bc);

        // Instantiate solver objects
        
        SolverStandardSOR solverStandardSOR(U.data(), width, height, "BasicSolver");
        SolverRedBlack solverStandardRedBlack(U.data(), width, height, "RedBlackSolver");

        // Determine which solver to run based on command-line arguments
        // Usage: ./PDE_GPUSolver [boundary_conditions.json] [solver_type]
        // solver_type options: basic, shared, thrust, all

        std::string solverType = "all"; // Default to running all solvers
        if (argc > 2){
            solverType = argv[2];
        }  

        // Function to plot solutions
        auto plot_solution = [&](const std::string& solver_type, const std::string& filename){
            std::string command = "python \"" + script_path + "\" \"" + solver_type + "\" \"" + filename + "\"";
            int ret = system(command.c_str());
            if (ret != 0){
                std::cerr << "Error: Plotting " << filename << " failed.\n";
            }
        };
        // 'basic_cpu', 'red_black_cpu'

        // Run the selected solver(s)
        if (solverType == "basic_cpu" || solverType == "all") {
            std::cout << "[Main] Running Basic Solver...\n";
            solverStandardSOR.solve();
            fs::path solution_basic_path = fs::path(project_dir) / "solutions" / "solution_basic_cpu.csv";
            std::string solution_basic_cpu = solution_basic_path.string();
            solverStandardSOR.exportSolution(solution_basic_cpu);
            plot_solution("basic_cpu", solution_basic_cpu);
            if (solverType == "all") {
                initializeGrid(U.data(), width, height, bc); // Re-initialize for next solver
            }
        }

        if (solverType == "red_black_cpu" || solverType == "all") {
           std::cout << "[Main] Running red and black sor Solver...\n";
            solverStandardRedBlack.solve();
           fs::path solution_red_black_cpu_path = fs::path(project_dir) / "solutions" / "solution_red_black_cpu.csv";
            std::string solution_red_black_cpu = solution_red_black_cpu_path.string();
            solverStandardRedBlack.exportSolution(solution_red_black_cpu);
            plot_solution("red_black_cpu", solution_red_black_cpu);
            if (solverType == "all") {
                initializeGrid(U.data(), width, height, bc);  // Re-initialize for next solver
            }
        }

    }
    catch(const std::exception& e){
        std::cerr << "Exception during solver execution: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}













//int main() {
    // Declare and initialize the main grid
//    std::vector<std::vector<double>> grid(M, std::vector<double>(N, 0.0));

    // Initialize the grid with boundary conditions
//    initializeGrid(grid);

    // Time the Standard SOR method
//    std::vector<std::vector<double>> gridStandard = grid;  // Copy grid for standard SOR
//    double standardTime = timeSOR(updateStandardSOR, gridStandard);
//    cout << "Standard SOR took " << standardTime << " seconds.\n";

    // Time the Red-Black SOR method
//    std::vector<std::vector<double>> gridRedBlack = grid;  // Copy grid for red-black SOR
//    double redBlackTime = timeSOR(updateRedBlackSOR, gridRedBlack);
//    cout << "Red-Black SOR took " << redBlackTime << " seconds.\n";

    // Validate the Red-Black SOR solution
//    validateSolution(gridRedBlack);

    // Export the solved Standard SOR solution and plot
//    exportSolutionAndPlot(gridStandard, "cpu_standard_sor_solution.txt", "Standard SOR Solution");

    // Export the solved Red-Black SOR solution and plot
//    exportSolutionAndPlot(gridRedBlack, "cpu_red_black_sor_solution.txt", "Red and Black SOR Solution");

//    return 0;
//}


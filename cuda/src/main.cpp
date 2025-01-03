// cuda/src/main.cpp

#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "solver_base.h"

#include "solver_basic.h"
#include "solver_shared.h"
#include "solver_thrust.h"

#include "utilities.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <filesystem> // C++17
#include <vector>
#include <cstdlib> // For system()
#include<unistd.h>
#include<limits.h>
#include<libgen.h>
#include <chrono>

namespace fs = std::filesystem;

// Function to check if a command exists
bool CommandExists(const std::string& cmd) {
    std::string check = "which " + cmd + " > /dev/null 2>&1";
    return (system(check.c_str()) == 0);
}

// Function to get the python command
std::string getPythonCommand(){
    if (CommandExists("python3")){
        return "python3";
    }else if (CommandExists("python")){
        return "python";
    }
    else{
        throw std::runtime_error("No Python interpreter found.");
    }
}


// Function to get the executable path
std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count == -1) {
        throw std::runtime_error("Unable to determine executable path.");
    }
    return std::string(result, count);
}


// Function to get the project directory
std::string getProjectDir() {
    std::string exePath = getExecutablePath();
    fs::path exeDir = fs::path(exePath).parent_path(); // Directory containing the executable

    // Adjust the number of parent_path() calls based on your project structure
    // For example, if executable is in build_cpu/cpu/, and project root is GPU-Laplacian-Solver/
    fs::path projectDir = exeDir.parent_path().parent_path(); // Two levels up

    return projectDir.string();
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

        // Allocate Unified Memory for the grid
        double *U;
        CUDA_CHECK_ERROR(cudaMallocManaged(&U, width * height * sizeof(double)));

        // Initialize the grid with boundary conditions
        initializeGrid(U, width, height, bc);

        // Instantiate solver objects
        SolverBasic solverBasic(U, width, height, "BasicSolver");
        SolverShared solverShared(U, width, height, "SharedMemorySolver");
        SolverThrust solverThrust(U, width, height, "ThrustSolver");

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

        // Run the selected solver(s)
        if (solverType == "basic_cuda" || solverType == "all") {
            std::cout << "[Main] Running Basic Solver...\n";
            solverBasic.solve();
            fs::path solution_basic_path = fs::path(project_dir) / "solutions" / "solution_basic_cuda.csv";
            std::string solution_basic = solution_basic_path.string();
            solverBasic.exportSolution(solution_basic);
            //plot_solution("basic_cuda", solution_basic);
            if (solverType == "all") {
                initializeGrid(U, width, height, bc); // Re-initialize for next solver
            }
        }

        if (solverType == "shared" || solverType == "all") {
            std::cout << "[Main] Running Shared Memory Solver...\n";
            solverShared.solve();
            fs::path solution_shared_path = fs::path(project_dir) / "solutions" / "solution_shared.csv";
            std::string solution_shared = solution_shared_path.string();
            solverShared.exportSolution(solution_shared);
            //plot_solution("shared", solution_shared);
            if (solverType == "all") {
                initializeGrid(U, width, height, bc); // Re-initialize for next solver
            }
        }

        if (solverType == "thrust" || solverType == "all") {
            std::cout << "[Main] Running Thrust Optimized Solver...\n";
            solverThrust.solve();
            fs::path solution_thrust_path = fs::path(project_dir) / "solutions" / "solution_thrust.csv";
            std::string solution_thrust = solution_thrust_path.string();
            solverThrust.exportSolution(solution_thrust);
            //plot_solution("thrust", solution_thrust);
        }

        // Free Unified Memory
        CUDA_CHECK_ERROR(cudaFree(U));

    }
    catch(const std::exception& e){
        std::cerr << "Exception during solver execution: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}



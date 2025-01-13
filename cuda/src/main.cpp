// cuda/src/main.cpp

#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "solver_base.h"

#include "solver_basic.h"
#include "solver_shared.h"
#include "solver_thrust.h"

#include "utilities.h"
#include "solution_export.h"  

#include <cstdlib>
#include <iostream>
#include <string>
#include <filesystem> // C++17
#include <vector>
#include <cstdlib> // For system()
#include <limits.h>
#include <chrono>

#ifdef __unix__
#include <libgen.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif



namespace fs = std::filesystem;

// Function to check if a command exists
#ifdef _WIN32
bool CommandExists(const std::string& cmd) {
    std::string check = "where " + cmd + " >nul 2>&1";
    return (system(check.c_str()) == 0);
}
#else
bool CommandExists(const std::string& cmd) {
    std::string check = "which " + cmd + " >/dev/null 2>&1";
    return (system(check.c_str()) == 0);
}
#endif

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

#ifdef _WIN32
std::string getExecutablePath() {
    char result[MAX_PATH];
    DWORD length = GetModuleFileNameA(nullptr, result, MAX_PATH);
    if (length == 0 || length == MAX_PATH) {
        throw std::runtime_error("Unable to determine executable path on Windows.");
    }
    return std::string(result, length);
}
#else
std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count == -1) {
        throw std::runtime_error("Unable to determine executable path on Linux.");
    }
    return std::string(result, count);
}
#endif


// Function to get the project directory
std::string getProjectDir() {
    std::string exePath = getExecutablePath();
    fs::path exeDir = fs::path(exePath).parent_path(); // Directory containing the executable

    // Adjust the number of parent_path() calls based on your project structure
    // For example, if executable is in build_cpu/cpu/, and project root is GPU-Laplacian-Solver/
    fs::path projectDir = exeDir.parent_path().parent_path().parent_path(); // Two levels up

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

        // Allocate Memory for the grid

        // Create host array to store the grid
        std::vector<double> U_host(width * height, 0.0);

        // Initialize the grid with boundary conditions
        initializeGrid(U_host.data(), width, height, bc);


        // Allocate device memory
        double *d_U = nullptr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_U, width * height * sizeof(double)));

        // 4) Copy host array to device array
        CUDA_CHECK_ERROR(cudaMemcpy(d_U, U_host.data(),
                                    width * height * sizeof(double),
                                    cudaMemcpyHostToDevice));

        // Instantiate solver objects
        SolverBasic solverBasic(d_U, width, height, "BasicSolver");
        SolverShared solverShared(d_U, width, height, "SharedMemorySolver");
        SolverThrust solverThrust(d_U, width, height, "ThrustSolver");

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
        // 8) Run the selected solver(s)
        //    Note: Each solver modifies d_U in place.

        // ------------------------------------------------------------
        // Basic CUDA solver
        // ------------------------------------------------------------
        if (solverType == "basic_cuda" || solverType == "all") {
            std::cout << "[Main] Running Basic Solver...\n";
            
            solverBasic.solve(); // This uses d_U on the device

            // (Optional) If you need to see or save final data on the CPU,
            // copy device -> host now:
            CUDA_CHECK_ERROR(cudaMemcpy(U_host.data(), d_U,
                                        width * height * sizeof(double),
                                        cudaMemcpyDeviceToHost));

            // Example: Export from host to CSV
            fs::path solution_basic_path = fs::path(project_dir) / "solutions" / "solution_basic_cuda.csv";
            //solverBasic.exportSolution(solution_basic_path.string());

            exportSolutionToCSV(solverBasic.getDevicePtr(),
                                width,
                                height,
                                solution_basic_path.string(),
                                solverBasic.getName());

            plot_solution("basic_cuda", solution_basic_path.string());

            // Re-initialize device array for next solver if user wants "all"
            if (solverType == "all") {
                // Re-init host array
                initializeGrid(U_host.data(), width, height, bc);
                // Copy host array to device array again
                CUDA_CHECK_ERROR(cudaMemcpy(d_U, U_host.data(),
                                            width * height * sizeof(double),
                                            cudaMemcpyHostToDevice));
            }
        }

        // ------------------------------------------------------------
        // Shared Memory Solver
        // ------------------------------------------------------------
        if (solverType == "shared" || solverType == "all") {
            std::cout << "[Main] Running Shared Memory Solver...\n";
            
            solverShared.solve(); // modifies d_U

            // Copy back if you want to see it on host or export
            CUDA_CHECK_ERROR(cudaMemcpy(U_host.data(), d_U,
                                        width * height * sizeof(double),
                                        cudaMemcpyDeviceToHost));

            fs::path solution_shared_path = fs::path(project_dir) / "solutions" / "solution_shared.csv";
            //solverShared.exportSolution(solution_shared_path.string());
            exportSolutionToCSV(solverShared.getDevicePtr(),
                                width,
                                height,
                                solution_shared_path.string(),
                                solverShared.getName());
            plot_solution("shared", solution_shared_path.string());

            // If "all", re-init for next solver
            if (solverType == "all") {
                initializeGrid(U_host.data(), width, height, bc);
                CUDA_CHECK_ERROR(cudaMemcpy(d_U, U_host.data(),
                                            width * height * sizeof(double),
                                            cudaMemcpyHostToDevice));
            }
        }

        // ------------------------------------------------------------
        // Thrust Solver
        // ------------------------------------------------------------
        if (solverType == "thrust" || solverType == "all") {
            std::cout << "[Main] Running Thrust Optimized Solver...\n";
            
            solverThrust.solve(); // modifies d_U

            // Copy back if needed
            CUDA_CHECK_ERROR(cudaMemcpy(U_host.data(), d_U,
                                        width * height * sizeof(double),
                                        cudaMemcpyDeviceToHost));

            fs::path solution_thrust_path = fs::path(project_dir) / "solutions" / "solution_thrust.csv";
            //solverThrust.exportSolution(solution_thrust_path.string());
            exportSolutionToCSV(solverThrust.getDevicePtr(),
                                width,
                                height,
                                solution_thrust_path.string(),
                                solverThrust.getName());
            plot_solution("thrust", solution_thrust_path.string());
        }

        // 9) Free device memory
        CUDA_CHECK_ERROR(cudaFree(d_U));
    }
    catch(const std::exception& e){
        std::cerr << "Exception during solver execution: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}


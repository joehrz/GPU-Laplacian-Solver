// cpu/src/main.cpp


#include "boundary_conditions.h"      // Centralized
#include "grid_initialization.h"     // Centralized
#include "solver_base.h"             // Centralized
#include "laplace_analytical_solution.h"        // Include concrete analytical solution

#include "solver_basic.h"            // CPU-specific
#include "solver_red_black.h"        // CPU-specific


#include <iostream>
#include <vector>
#include <chrono>
#include <string>  // For std::string
#include <filesystem>  // Add this line
#include <cstdlib> // For system()
#include<unistd.h>
#include<limits.h>
#include<libgen.h>
#include <chrono>

using namespace std;
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


double computeL2Error(const std::vector<double>& numeric,
                      const std::vector<double>& exact,
                      int width, int height,
                      bool skipZeros = true
)
{
    double sumSquaredError = 0.0;
    int count = 0;
    for (int j = 1; j < height - 1; ++j) {
        for (int i = 1; i < width - 1; ++i) {
            double e = exact[j * width + i];
            double u = numeric[j * width + i];

            // If skipZeros == true, ignore exact=0 region
            // If skipZeros == false, count all interior points
            if (!skipZeros || std::abs(e) > 1e-15) {
                double diff = (u - e);
                sumSquaredError += diff * diff;
                ++count;
            }
        }
    }
    if (count == 0) {
        std::cerr << "Warning: zero interior points were counted.\n";
        return 0.0;
    }
    return std::sqrt(sumSquaredError / count);
}


int main(int argc, char* argv[]){
    std::string project_dir;
    std::string script_path;
    std::string bc_file;

    try {
        project_dir = getProjectDir(); // Default is 3 levels up
        //std::cout << "Project Directory: " << project_dir << "\n";

        // Ensure solutions directory exists
        fs::path solutions_dir = fs::path(project_dir) / "solutions";
        if (!fs::exists(solutions_dir)) {
            fs::create_directories(solutions_dir);
            //std::cout << "Created solutions directory: " << solutions_dir << "\n";
        }

        // Constructing paths relative to the project directory
        fs::path bc_file_path = fs::path(project_dir) / "boundary_conditions" / "boundary_conditions.json";
        bc_file = bc_file_path.string();
        //std::cout << "Boundary Conditions File Path: " << bc_file << "\n";

        fs::path script_path_fs = fs::path(project_dir) / "scripts" / "plot_solution.py";
        script_path = script_path_fs.string();
        //std::cout << "Plotting Solutions File Path: " << script_path << "\n";
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
        const int width = 101;
        const int height = 101;
        
        // Instantiate the analytical solution
        UniversalFourierSolution analytical(bc.left, bc.right, bc.top, bc.bottom, /* n_max = */ 25);
        std::vector<double> U_exact = analytical.compute(width, height);
        
        // Using std::vector for grid
        std::vector<double> U(width * height, 0.0);
        // Initialize the grid with boundary conditions       
        initializeGrid(U.data(), width, height, bc);

        // Instantiate solver objects
        SolverStandardSOR solverStandardSOR(U.data(), width, height, "BasicSolver");
        SolverRedBlack solverStandardRedBlack(U.data(), width, height, "RedBlackSolver");

        // Determine which solver to run based on command-line arguments
        std::string solverType = "all"; // Default to running all solvers
        if (argc > 2){
            solverType = argv[2];
        }  

        // Get the Python command
        std::string python_cmd = getPythonCommand();

        // Function to plot solutions
        auto plot_solution = [&](const std::string& solver_type, const std::string& filename){
            std::string command = python_cmd + " \"" + script_path + "\" \"" + solver_type + "\" \"" + filename + "\"";
            int ret = system(command.c_str());
            if (ret != 0){
                std::cerr << "Error: Plotting " << filename << " failed.\n";
            }
        };
        // 'basic_cpu', 'red_black_cpu'


        // 6a) Run Basic SOR solver
        if (solverType == "basic_cpu" || solverType == "all") {
            std::cout << "[Main] Running Basic Solver...\n";


            auto start = std::chrono::high_resolution_clock::now();
            solverStandardSOR.solve();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Standard SOR took " << duration.count() << " seconds.\n";



            // Compute L2 Error
            double L2Error = computeL2Error(U, U_exact, width, height, /*skipZeros=*/true);
            std::cout << "L2 Norm of Error (Basic SOR): " << L2Error << std::endl;

            // Export solution
            fs::path solution_basic_path = fs::path(project_dir) / "solutions" / "solution_basic_cpu.csv";
            solverStandardSOR.exportSolution(solution_basic_path.string());

            // (Optional) Plot
            // plot_solution("basic_cpu", solution_basic_path.string());

            // Re-initialize grid if you want to reuse the same "U" array for next solver
            if (solverType == "all") {
                initializeGrid(U.data(), width, height, bc);
            }
        }

        // 6b) Run Red-Black SOR solver
        if (solverType == "red_black_cpu" || solverType == "all") {
            std::cout << "[Main] Running Red-Black SOR Solver...\n";
            
            auto start = std::chrono::high_resolution_clock::now();
            solverStandardRedBlack.solve();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Standard SOR took " << duration.count() << " seconds.\n";


            // Compute L2 Error
            double L2Error = computeL2Error(U, U_exact, width, height, /*skipZeros=*/true);
            std::cout << "L2 Norm of Error (Red-Black SOR): " << L2Error << std::endl;

            // Export solution
            fs::path solution_rb_path = fs::path(project_dir) / "solutions" / "solution_red_black_cpu.csv";
            solverStandardRedBlack.exportSolution(solution_rb_path.string());

            // (Optional) Plot
            // plot_solution("red_black_cpu", solution_rb_path.string());
        }

    }
    catch(const std::exception& e) {
        std::cerr << "Exception during solver execution: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}














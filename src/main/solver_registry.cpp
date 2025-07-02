// src/main/solver_registry.cpp

#include "solver_registry.hpp"
#include <stdexcept>

#ifdef BUILD_CPU
#include "solver_standard_sor.hpp"
#include "solver_red_black.hpp"
#endif

// Conditionally include CUDA solver headers
#ifdef SOLVER_ENABLE_CUDA
#include "solver_basic_cuda.hpp"
#include "solver_shared_cuda.hpp"
#endif
// Define convenient type aliases
using Ptr = std::unique_ptr<Solver>;
using Vec = std::vector<Ptr>;

// Helper function to simplify adding a solver to the vector
static void add(Vec& v, Ptr&& p){
    v.emplace_back(std::move(p));
}

// The factor function implementation
Vec make_solvers(const std::string& which, float* host, int W, int H) {
    Vec v;

    // Conditionally compile the CPU solver creation code
    #ifdef BUILD_CPU
    if (which == "basic_cpu" || which == "all") {
        add(v, std::make_unique<SolverStandardSOR>(host, W, H, "BasicSOR_CPU"));
    }
    if (which == "red_black_cpu" || which == "all") {
        add(v, std::make_unique<SolverRedBlack>(host, W, H, "RedBlackSOR_CPU"));
    }
    #endif

    #ifdef SOLVER_ENABLE_CUDA
    if (which == "basic_cuda" || which == "all") {
        add(v, std::make_unique<SolverBasicCUDA>(host, W, H, "BasicCUDA"));
    }
    if (which == "shared_cuda" || which == "all") {
        add(v, std::make_unique<SolverSharedMemCUDA>(host, W, H, "SharedMemCUDA"));
    }
    #endif

    if (v.empty()) {
        throw std::runtime_error("Unknown or no solvers requested for tag: " + which);
    }
    return v;
}
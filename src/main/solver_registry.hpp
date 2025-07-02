// src/main/solver_registry.hpp

#pragma once
#include "solver_base.h"
#include <memory>
#include <string>
#include <vector>

// This function acts as a factory, creating a vector of solver objects
std::vector<std::unique_ptr<Solver>>
make_solvers(const std::string& which, float* host_grif, int W, int H);
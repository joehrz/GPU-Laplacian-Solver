// src/common/include/solver_base.h

#pragma once
#include "simulation_config.h" // For SimulationParameters
#include <string>

class Solver {
protected:
    std::string name;
    float* grid_ptr;
    int width, height;

public:
    // Constructor
    Solver(float* grid, int w, int h, const std::string& n);
    
    // Virtual destructor
    virtual ~Solver() = default;

    // Pure virtual function that all solvers implement
    virtual void solve(const SimulationParameters& params) = 0;

    // Virtual functions to check location of data (defaults to CPU)
    virtual bool isOnDevice() const;
    virtual float* deviceData();

    // Getter for the solver's name
    const std::string& getName() const;
};
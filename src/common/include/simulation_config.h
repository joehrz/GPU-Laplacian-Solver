// src/common/include/simulation_config.h

#pragma once
#include "boundary_conditions.h"
#include <string>

struct SimulationParameters {
    int   width          = 256;
    int   height         = 256;
    float tolerance      = 1e-5f;
    int   max_iterations = 10000;
    float omega          = 1.9f;
};

struct FullConfig{
    BoundaryConditions bc;
    SimulationParameters sim_params;
};

FullConfig loadConfiguration(const std::string& filename);


// common/include/simulation_config.h
#ifndef SIMULATION_CONFIG_H
#define SIMULATION_CONFIG_H

#include "boundary_conditions.h"
#include <string>

struct SimulationParameters {
    int width;
    int height;
    float tolerance;
    int max_iterations;
    float omega;
};

struct FullConfig {
    BoundaryConditions bc;
    SimulationParameters sim_params;
};

FullConfig loadConfiguration(const std::string& filename);

#endif // SIMULATION_CONFIG_H



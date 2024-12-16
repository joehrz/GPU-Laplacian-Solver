// cpu/src/boundary_conditions.cpp

#include "boundary_conditions.h"
#include <nlohmann/json.hpp> // nlohmann/json library
#include <fstream>
#include <iostream>

// Use the nlohmann::json namespace for convenience
using json = nlohmann::json;

// Function to load boundary conditions from a JSON file
BoundaryConditions loadBoundaryConditions(const std::string& filename) {
    BoundaryConditions bc;
    
    // Open the JSON file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open boundary conditions file: " << filename << std::endl;
        std::cerr << "Using default boundary conditions." << std::endl;
        // Set default values
        bc.left = 0.0;
        bc.right = 0.0;
        bc.top = 0.0;
        bc.bottom = 100.0;
        return bc;
    }

    try {
        // Parse the JSON content
        json j;
        file >> j;

        // Extract boundary condition values with default fallbacks
        bc.left = j.value("left", 0.0);
        bc.right = j.value("right", 0.0);
        bc.top = j.value("top", 0.0);
        bc.bottom = j.value("bottom", 100.0);
    }
    catch (const json::parse_error& e) {
        std::cerr << "JSON Parse Error: " << e.what() << std::endl;
        std::cerr << "Using default boundary conditions." << std::endl;
        // Set default values in case of parse error
        bc.left = 0.0;
        bc.right = 0.0;
        bc.top = 0.0;
        bc.bottom = 100.0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Using default boundary conditions." << std::endl;
        // Set default values in case of any other exceptions
        bc.left = 0.0;
        bc.right = 0.0;
        bc.top = 0.0;
        bc.bottom = 100.0;
    }

    std::cout << "Boundary conditions loaded from " << filename << std::endl;
    return bc;
}

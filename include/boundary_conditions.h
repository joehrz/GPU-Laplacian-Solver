// boundary_conditions.h
#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <string>

struct BoundaryConditions{
    float left;
    float right;
    float top;
    float bottom;
};

BoundaryConditions loadBoundaryConditions(const std::string& filename);

#endif // BOUNDARY_CONDITIONS_H
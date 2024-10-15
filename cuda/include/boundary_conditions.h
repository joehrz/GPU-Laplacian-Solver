// boundary_conditions.h
#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <string>

struct BoundaryConditions{
    double left;
    double right;
    double top;
    double bottom;
};

BoundaryConditions loadBoundaryConditions(const std::string& filename);

#endif // BOUNDARY_CONDITIONS_H
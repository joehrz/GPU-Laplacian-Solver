// grid_initialization.h
#ifndef GRID_INITIALIZATION_H
#define GRID_INITIALIZATION_H

#include "boundary_conditions.h" // Include the header where BoundaryConditions is defined

void initializeGrid(float *U, int width, int height, const BoundaryConditions& bc);

#endif // GRID_INITIALIZATION_H
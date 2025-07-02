// src/common/include/grid_initialization.h

#pragma once
#include "boundary_conditions.h"

void initializeGrid(float* U, int width, int height,
                    const BoundaryConditions& bc);

                    
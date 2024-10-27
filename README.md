# PDE-GPU-Solver

**PDE-GPU-Solver** is a C++ and CUDA project that implements the Successive Over-Relaxation (SOR) and Red-Black SOR methods for solving Laplace's equation in two dimensions. The project includes both CPU and GPU implementations, validation against analytical solutions, and plotting functionalities.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

PDE-GPU-Solver/
│
├── cpu
│   ├── src
│   │   ├── main.cpp
│   │   ├── red_black_sor.cpp
│   │   ├── sor_methods.cpp
│   │   └── standard_sor.cpp
│   │          
│   ├── tests  
│   ├── CMakeLists.txt       
│   └── README.md
│
├── cuda
│   ├── data
│   │   ├── boundary_conditions.json
│   │   └── solutions
│   ├── external
│   │   ├── nlohmann
│   │       ├── json_fwd
│   │       └── json
│   │
│   ├── include
│   │   ├── boundary_conditions.h
│   │   ├── grid_initialization.h
│   │   ├── solver_base.h
│   │   ├── solver_basic.h
│   │   ├── solver_shared.h
│   │   ├── solver_thrust.h
│   │   └── utilities.h
│   │
│   ├── scripts
│   │   └── plot_solution.py
│   │             
│   ├── src
│   │   ├── boundary_conditions.cpp
│   │   ├── grid_initialization.cpp
│   │   ├── main.cpp
│   │   ├── solver_basic.cu
│   │   ├── solver_shared.cu
│   │   ├── solver_thrust.cu
│   │   └── sor_red_black.cu
│   │            
│   ├── tests
│   │   ├── test_solver_basic.cpp
│   │   ├── test_solver_shared.cpp
│   │   └── test_solver_thrust.cpp
│   │              
│   └──CMakeLists.txt
|
│
├── examples             
├── CMakeLists.txt      
├── LICENSE
├── algorithm.md             
└── README.md  


## Features

- **CPU Implementation**: The `cpu/` directory contains the CPU version of the SOR and Red-Black SOR methods.
- **GPU Implementation**: The `cuda/` directory contains the GPU-accelerated versions using CUDA.
- **Analytical Solution**: Provides analytical solutions for validation purposes.
- **Plotting**: Includes Python scripts for visualizing the results.
- **Testing**: Contains test suites to validate the correctness of the implementations.

## Installation

### Prerequisites

- **C++ Compiler**: Required for compiling the CPU version (e.g., `g++` or `clang++`).
- **CUDA Toolkit**: Required for compiling the GPU version.
- **CMake**: For build configuration.
- **Python 3**: For running plotting scripts (`plot.py`).
- **Gnuplot**: If using Gnuplot for plotting results.

### Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/PDE-GPU-Solver.git
   cd PDE-GPU-Solver

# GPU-Laplacian-Solver
> **Note**: This project is still a **work in progress**. 

**GPU-Laplacian-Solver** is a C++/CUDA project that implements the **Successive Over-Relaxation (SOR)** and **Red-Black SOR** methods to solve **Laplace’s equation** in two dimensions. It provides both CPU and GPU implementations, can validate results against an analytical solution, and includes Python-based plotting tools.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Building CPU Only](#building-cpu-only)
  - [Building GPU Only](#building-gpu-only)
  - [Building Both CPU and GPU](#building-both-cpu-and-gpu)
- [Project Structure](#project-structure)

---

## Features

- **CPU Implementation**  
  - \[`cpu/`\]: Implements SOR and Red-Black SOR on the CPU.  

- **GPU Implementation**  
  - \[`cuda/`\]: Implements CUDA-based solvers, including basic SOR kernels, shared-memory optimizations, and thrust-based methods.

- **Analytical Solution**  
  - Provides a Fourier-based solution for Laplace’s equation for validation.

- **Plotting**  
  - Python script (\[`scripts/plot_solution.py`\]) to visualize the 2D solution as a heatmap or surface plot.

- **Testing**  
  - Contains basic test setups (especially in the GPU subfolder) to validate solver correctness.

---


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
- **CMake**: For build configuration (version 3.20 or higher recommended).
- **Python 3**: For running plotting script.

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/joehrz/GPU-Laplacian-Solver.git
   cd PDE-GPU-Solver

2. **Building**

   ```bash
   mkdir build
   cd build
   cmake .. -DBUILD_CPU=ON -DBUILD_CUDA=ON
   cmake --build . --config Release



## Project Structure

```plaintext
GPU-Laplacian-Solver/
│
├── cpu
│   ├── include
│   │   ├── boundary_conditions.h
│   │   ├── config.h
│   │   ├── grid_initialization.h
│   │   ├── solver_base.h
│   │   ├── solver_basic.h
│   │   └── solver_red_black.h
│   ├── src
│   │   ├── main.cpp
│   │   ├── red_black_sor.cpp
│   │   ├── standard_sor.cpp
│   │   ├── boundary_conditions.cpp
│   │   └── grid_initialization.cpp
│   └── CMakeLists.txt
│
├── cuda
│   ├── include
│   │   ├── boundary_conditions.h
│   │   ├── grid_initialization.h
│   │   ├── solver_base.h
│   │   ├── solver_basic.h
│   │   ├── solver_shared.h
│   │   ├── solver_thrust.h
│   │   └── utilities.h
│   ├── src
│   │   ├── boundary_conditions.cpp
│   │   ├── grid_initialization.cpp
│   │   ├── main.cpp
│   │   ├── solver_basic.cu
│   │   ├── solver_shared.cu
│   │   ├── solver_thrust.cu
│   │   └── sor_red_black.cu
│   ├── tests
│   │   ├── test_solver_basic.cpp
│   │   ├── test_solver_shared.cpp
│   │   └── test_solver_thrust.cpp
│   └── CMakeLists.txt
│
├── scripts
│   └── plot_solution.py
│
├── boundary_conditions
│   └── boundary_conditions.json    (example or default input)
│
├── CMakeLists.txt                  (Top-level)
├── LICENSE
├── algorithm.md
└── README.md

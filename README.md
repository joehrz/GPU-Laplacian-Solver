# PDE-GPU-Solver

PDE-GPU-Solver is a C++ project that implements the Successive Over-Relaxation (SOR) and Red-Black SOR methods for solving Laplace's equation in two dimensions. The project also includes functionality for validating the numerical solutions against an analytical solution, as well as plotting the results using Gnuplot.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project is organized as follows:

## Features

- **Standard SOR Method**: Implemented to solve the Laplace equation using successive over-relaxation.
- **Red-Black SOR Method**: An optimized variant of the SOR method that uses a checkerboard update scheme.
- **Analytical Solution**: Computes the analytical solution for validation against the numerical methods.
- **Performance Measurement**: Measures the execution time of the SOR methods.
- **Plotting**: Generates plots of the numerical and analytical solutions using Gnuplot.

## Installation

To build the project, you will need a C++ compiler (e.g., g++) and Gnuplot for plotting.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/PDE-GPU-Solver.git
   cd PDE-GPU-Solver

## Installation
```bash
git clone https://github.com/Dxxc/GPU-Laplacian-Solver.git
cd PDE-Solver
mkdir build
cd build
cmake ..
make

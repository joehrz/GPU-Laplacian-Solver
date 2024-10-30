# PDE-GPU-Solver/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ========================================================================
# Project Declaration
# ========================================================================

project(PDE-GPU-Solver VERSION 1.0 LANGUAGES CXX CUDA)

# ========================================================================
# Compiler Standards
# ========================================================================

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# ========================================================================
# Build Options
# ========================================================================

option(BUILD_CPU "Build the CPU version of the solver" ON)
option(BUILD_CUDA "Build the CUDA version of the solver" ON)

# ========================================================================
# Enable Testing
# ========================================================================

include(CTest)
enable_testing()

# ========================================================================
# Add Subdirectories
# ========================================================================

if(BUILD_CPU)
    add_subdirectory(cpu)
endif()

if(BUILD_CUDA)
    add_subdirectory(cuda)
endif()

# ========================================================================
# Installation (Optional)
# ========================================================================

# Uncomment and configure the following lines if you wish to enable installation
# install(DIRECTORY cuda/data/ DESTINATION data)
# install(TARGETS PDE_Solver-CPU PDE_Solver-GPU
#         RUNTIME DESTINATION bin)

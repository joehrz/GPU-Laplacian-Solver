# GPU-Laplacian-Solver/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ========================================================================
# Project Declaration
# ========================================================================

project(PDE_GPUSolver VERSION 1.0 LANGUAGES CXX CUDA)

# ========================================================================
# Compiler Standards
# ========================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ========================================================================
# Build Options
# ========================================================================

option(BUILD_CPU "Build the CPU version of the solver" ON)
option(BUILD_CUDA "Build the CUDA version of the solver" ON)

# Find the CUDA toolkit early so CUDA::cudart is a known target
if (BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ========================================================================
# External Dependencies
# ========================================================================

# Use FetchContent to download nlohmann_json
include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# ========================================================================
# Common Library (SolverCommon)
#
# This section defines a modern, two-part library.
# 1. An INTERFACE library that specifies the "usage requirements" (e.g., include paths).
# 2. A STATIC library that contains the actual compiled code.
# ========================================================================

# Part 1: Define an INTERFACE library to specify the "usage requirements" for any solver.
add_library(SolverCommon INTERFACE)

# Part 2: Specify the include directories needed by any library/executable that USES SolverCommon.
# This replaces the old global include_directories().
target_include_directories(SolverCommon INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
    # Make nlohmann::json headers available to all consumers of SolverCommon
    $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
)

# Part 3: Add CUDA-specific requirements to the interface if building with CUDA.
if(BUILD_CUDA)
    target_compile_definitions(SolverCommon INTERFACE SOLVER_ENABLE_CUDA)
    target_link_libraries(SolverCommon INTERFACE CUDA::cudart)
endif()

# Part 4: Create a separate STATIC library for the actual compiled common source files.
add_library(SolverCommonSources STATIC
    common/src/config_loader.cpp
    common/src/grid_initialization.cpp
    common/src/solution_export.cpp
)

# Part 5: Link the source library to the interface. This means any target linking
# to SolverCommonSources will automatically get the include paths and definitions from SolverCommon.
target_link_libraries(SolverCommonSources PUBLIC SolverCommon)

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
# End of CMakeLists.txt
# ========================================================================
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
set(CMAKE_CUDA_EXTENSIONS OFF) # Optional: turn off if you need GNU extensions with CUDA

# ========================================================================
# Build Options
# ========================================================================

option(BUILD_CPU "Build the CPU version of the solver" ON)
option(BUILD_CUDA "Build the CUDA version of the solver" ON)


# ------------------------------------------------------------------------
# Find the CUDA toolkit early so CUDA::cudart is a known target
# ------------------------------------------------------------------------
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
  GIT_TAG v3.11.3 # Specify your desired version
)
FetchContent_MakeAvailable(json)



# ========================================================================
# Include Directories (Global)
# ========================================================================

# Add the central include directory and the new common include directory
include_directories(
    ${CMAKE_SOURCE_DIR}/include         # For headers like solver_base.h, boundary_conditions.h etc.
    ${CMAKE_SOURCE_DIR}/common/include  # For simulation_config.h
)

# ========================================================================
# Common Library (SolverCommon)
# ========================================================================

add_library(SolverCommon STATIC
    common/src/config_loader.cpp
    common/src/grid_initialization.cpp
    common/src/solution_export.cpp  
)

# Link SolverCommon against nlohmann_json as it's used by config_loader.cpp
# Making nlohmann_json PUBLIC here means targets linking SolverCommon also get nlohmann_json headers/properties.
target_link_libraries(SolverCommon PUBLIC nlohmann_json::nlohmann_json)

if (BUILD_CUDA)
    target_compile_definitions(SolverCommon
        PUBLIC SOLVER_ENABLE_CUDA)
    target_link_libraries(SolverCommon
        PUBLIC CUDA::cudart)
endif()

# Ensure SolverCommon is built with the project's C++ standard
set_target_properties(SolverCommon PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD})

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


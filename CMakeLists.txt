# GPU-Laplacian-Solver/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ========================================================================
# Project Declaration
# ========================================================================

project(PDE_GPUSolver VERSION 1.0 LANGUAGES CXX CUDA)

# ========================================================================
# Compiler Standards
# ========================================================================

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ========================================================================
# Build Options
# ========================================================================

option(BUILD_CPU "Build the CPU version of the solver" ON)
option(BUILD_CUDA "Build the CUDA version of the solver" ON)

# ========================================================================
# External Dependencies
# ========================================================================

# Use FetchContent to download nlohmann_json
include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2 # Specify the version you need
)

FetchContent_MakeAvailable(json)

# ========================================================================
# Include Directories
# ========================================================================

# Add the central include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

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
# install(TARGETS pde_solver_cpu PDE_Solver-GPU
#         RUNTIME DESTINATION bin)


# GPU-Laplacian-Solver/CMakeLists.txt

# =======================================================================================
#  GPU-Laplacian-Solver - Root CMakeLists.txt
# =======================================================================================

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------------------
# 1.  Project declaration
# ---------------------------------------------------------------------------------------
project(GPU_Laplacian_Solver VERSION 1.0 LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------------------
# 2.  Compiler standards
# ---------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------------------------------------------------------------------------
# 3.  Build options
# ---------------------------------------------------------------------------------------
option(BUILD_CPU   "Build the CPU version of the solver"   ON)
option(BUILD_CUDA  "Build the CUDA version of the solver"  ON)
option(BUILD_TESTS "Build unit tests with GoogleTest"      ON)

# Find the CUDA toolkit early so CUDA::cudart is a known target
if (BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------------------------------
# 4.  External dependencies fetched with FetchContent
# ---------------------------------------------------------------------------------------
include(FetchContent)

# nlohmann/json (header-only)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# GoogleTest will be fetched *only* if BUILD_TESTS is ON (see §7)

# ---------------------------------------------------------------------------------------
# 5.  “Common” interface and source libraries (always present)
# ---------------------------------------------------------------------------------------
# Part A – interface target that only carries include paths / defines
add_library(SolverCommon INTERFACE)

target_include_directories(SolverCommon INTERFACE
    ${CMAKE_SOURCE_DIR}/src/common/include
    $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
)

if (BUILD_CUDA)
    target_compile_definitions(SolverCommon INTERFACE SOLVER_ENABLE_CUDA)
    target_link_libraries  (SolverCommon INTERFACE CUDA::cudart)
endif()

# Part B – actual source files (STATIC) that implement shared functionality
add_library(SolverCommonSources STATIC
    src/common/src/config_loader.cpp
    src/common/src/grid_initialization.cpp
    src/common/src/laplace_analytical_solution.cpp
    src/common/src/paths.cpp
    src/common/src/python.cpp
    src/common/src/solution_export.cpp
    src/common/src/timers.cpp
    src/common/src/solver_base.cpp
)

target_link_libraries(SolverCommonSources PUBLIC SolverCommon)
target_compile_definitions(SolverCommonSources
    PUBLIC -DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if (BUILD_CPU)
    target_compile_definitions(SolverCommonSources INTERFACE -DBUILD_CPU)
endif()

# ---------------------------------------------------------------------------------------
# 6.  Solver libraries (must come *before* anything that links to them)
# ---------------------------------------------------------------------------------------
if (BUILD_CPU)
    add_subdirectory(src/cpu_solver)   # defines cpu_solver target
endif()

if (BUILD_CUDA)
    add_subdirectory(src/cuda_solver)  # defines cuda_solver target
endif()

# ---------------------------------------------------------------------------------------
# 7.  Main executable (optional, independent of tests)
# ---------------------------------------------------------------------------------------
add_subdirectory(src/main)

# ---------------------------------------------------------------------------------------
# 8.  Unit tests (added last so all libs already exist)
# ---------------------------------------------------------------------------------------
if (BUILD_TESTS)
    enable_testing()

    # Fetch GoogleTest
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    # Add the test suite directory (defines run_tests target)
    add_subdirectory(src/tests)
endif()

# =======================================================================================
#  End of file
# =======================================================================================
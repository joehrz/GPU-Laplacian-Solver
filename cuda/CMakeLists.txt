# CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ========================================================================
# Project Declaration
# ========================================================================

project(CUDA_SOR_RedBlack_Solver LANGUAGES CXX CUDA)

# ========================================================================
# CUDA Settings
# ========================================================================

# Use static CUDA runtime library
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Specify CUDA architectures based on your GPU's compute capability
set(CMAKE_CUDA_ARCHITECTURES native)

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
# Include Directories
# ========================================================================

# Define common include directories
set(PROJECT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# ========================================================================
# Find CUDA Toolkit
# ========================================================================

find_package(CUDAToolkit REQUIRED)

# Find cudadevrt.lib
find_library(CUDA_CUDADEVRT_LIBRARY
    NAMES cudadevrt.lib cudadevrt
    PATHS "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
    NO_DEFAULT_PATH
)

# ========================================================================
# Define Libraries
# ========================================================================

# --------------------------
# SharedLib: Common Sources
# --------------------------

add_library(SharedLib STATIC
    src/grid_initialization.cpp
    src/boundary_conditions.cpp
)

# Specify include directories for SharedLib
target_include_directories(SharedLib PUBLIC
    ${PROJECT_INCLUDE_DIRS}
)

# Apply compiler options to SharedLib
if (MSVC)
    target_compile_options(SharedLib PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
    )
else()
    target_compile_options(SharedLib PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
    )
endif()

# --------------------------
# SolverBasic: Basic GPU Solver
# --------------------------

add_library(SolverBasic STATIC
    src/solver_basic.cu
)

# Enable separable compilation and device symbol resolution
set_target_properties(SolverBasic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Specify include directories for SolverBasic
target_include_directories(SolverBasic PUBLIC
    ${PROJECT_INCLUDE_DIRS}
)

# Link SolverBasic with CUDA runtime
target_link_libraries(SolverBasic PUBLIC CUDA::cudart)

# --------------------------
# SolverShared: Shared Memory Optimized Solver
# --------------------------

add_library(SolverShared STATIC
    src/solver_shared.cu
)

# Enable separable compilation and device symbol resolution
set_target_properties(SolverShared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Specify include directories for SolverShared
target_include_directories(SolverShared PUBLIC
    ${PROJECT_INCLUDE_DIRS}
)

# Link SolverShared with CUDA runtime
target_link_libraries(SolverShared PUBLIC CUDA::cudart)

# --------------------------
# SolverThrust: Thrust Optimized Solver
# --------------------------

add_library(SolverThrust STATIC
    src/solver_thrust.cu
)

# Enable separable compilation and device symbol resolution
set_target_properties(SolverThrust PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Specify include directories for SolverThrust
target_include_directories(SolverThrust PUBLIC
    ${PROJECT_INCLUDE_DIRS}
)

# Link SolverThrust with CUDA runtime
target_link_libraries(SolverThrust PUBLIC CUDA::cudart)

# Add the extended lambda compile option for SolverThrust
target_compile_options(SolverThrust PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

# ========================================================================
# Define Main Executable
# ========================================================================

add_executable(PDE_Solver
    src/main.cpp
)

# Set properties for CUDA device symbol resolution and linker language
set_target_properties(PDE_Solver PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

# Link PDE_Solver with CUDA libraries, SharedLib, and cudadevrt.lib
target_link_libraries(PDE_Solver PRIVATE
    SolverBasic
    SolverShared
    SolverThrust
    SharedLib
    ${CUDA_CUDADEVRT_LIBRARY}
)

# Apply compiler options to PDE_Solver
if (MSVC)
    target_compile_options(PDE_Solver PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
    )
else()
    target_compile_options(PDE_Solver PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
    )
endif()

# ========================================================================
# Testing Configuration
# ========================================================================

enable_testing()

# Function to add solver tests
function(add_solver_test test_name test_src solver_lib)
    add_executable(${test_name} ${test_src})

    # Set properties for CUDA device symbol resolution and linker language
    set_target_properties(${test_name} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        LINKER_LANGUAGE CUDA
    )

    # Apply compiler options to test executable
    if (MSVC)
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
        )
    else()
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
        )
    endif()

    target_link_libraries(${test_name} PRIVATE
        ${solver_lib}
        SharedLib
        ${CUDA_CUDADEVRT_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add tests for each GPU solver
add_solver_test(test_solver_basic tests/test_solver_basic.cpp SolverBasic)
add_solver_test(test_solver_shared tests/test_solver_shared.cpp SolverShared)
add_solver_test(test_solver_thrust tests/test_solver_thrust.cpp SolverThrust)

# ========================================================================
# Custom Clean Target
# ========================================================================

add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
)


# ========================================================================
# Optional: Install Targets
# ========================================================================

# Uncomment the following lines if you want to enable installation
# install(TARGETS PDE_Solver
#         RUNTIME DESTINATION bin)

# install(DIRECTORY data/ DESTINATION data)

# ========================================================================
# End of CMakeLists.txt
# ========================================================================



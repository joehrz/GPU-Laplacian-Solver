# GPU-Laplacian-Solver/cuda/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)


# ========================================================================
# CUDA Settings
# ========================================================================

# Use static CUDA runtime library
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Specify CUDA architectures based on your GPU's compute capability
# You can set it to 'native' to automatically detect the architecture
#set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES 70)
# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)


# ========================================================================
# Define Libraries
# ========================================================================

# --------------------------
# SharedLib: Common Sources
# --------------------------

add_library(SharedLib STATIC
    src/grid_initialization.cpp
    src/boundary_conditions.cpp
    src/solution_export.cpp
)

# Specify include directories for SharedLib
target_include_directories(SharedLib PUBLIC
    ${CMAKE_SOURCE_DIR}/include  # Central include directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CUDA-specific include directory
    ${CUDAToolkit_INCLUDE_DIRS}          # <-- Add this line
)

# Link SharedLib with nlohmann_json
target_link_libraries(SharedLib PUBLIC nlohmann_json::nlohmann_json)

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
    ${CMAKE_SOURCE_DIR}/include  # Central include directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CUDA-specific include directory
)

# Link SolverBasic with CUDA runtime and nlohmann_json
target_link_libraries(SolverBasic PUBLIC CUDA::cudart nlohmann_json::nlohmann_json)

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
    ${CMAKE_SOURCE_DIR}/include  # Central include directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CUDA-specific include directory
)

# Link SolverShared with CUDA runtime and nlohmann_json
target_link_libraries(SolverShared PUBLIC CUDA::cudart nlohmann_json::nlohmann_json)

# ========================================================================
# Define Main Executable
# ========================================================================

add_executable(PDE_Solver
    src/main.cpp
)

# Now that PDE_Solver exists, we can set its include directories
target_include_directories(PDE_Solver PRIVATE
    ${CMAKE_SOURCE_DIR}/include  # Central include directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CUDA-specific include directory
)

# Set properties for CUDA device symbol resolution and linker language
set_target_properties(PDE_Solver PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/cuda/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/cuda/Debug
)

# Link PDE_Solver with CUDA libraries, SharedLib, and nlohmann_json
target_link_libraries(PDE_Solver PRIVATE
    SolverBasic
    SolverShared
    SharedLib
    nlohmann_json::nlohmann_json
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
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/cuda/Release
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/cuda/Debug
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
        nlohmann_json::nlohmann_json
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add tests for each GPU solver
add_solver_test(test_solver_basic tests/test_solver_basic.cpp SolverBasic)
add_solver_test(test_solver_shared tests/test_solver_shared.cpp SolverShared)

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
#         RUNTIME DESTINATION bin/cuda)
# install(DIRECTORY data/ DESTINATION data/cuda)

# ========================================================================
# End of CMakeLists.txt
# ========================================================================






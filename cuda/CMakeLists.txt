# GPU-Laplacian-Solver/cuda/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ========================================================================
# CUDA Settings
# ========================================================================

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES 70) 

# --------------------------
# SolverBasic: Basic GPU Solver
# --------------------------
add_library(SolverBasic STATIC
    src/solver_basic.cu
)
set_target_properties(SolverBasic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(SolverBasic PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CUDA-specific include directory
)
target_link_libraries(SolverBasic PUBLIC CUDA::cudart SolverCommon) # Link SolverCommon

# --------------------------
# SolverShared: Shared Memory Optimized Solver
# --------------------------
add_library(SolverShared STATIC
    src/solver_shared.cu
)
set_target_properties(SolverShared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(SolverShared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CUDA-specific include directory
)
target_link_libraries(SolverShared PUBLIC CUDA::cudart SolverCommon) # Link SolverCommon

# ========================================================================
# Define Main Executable (PDE_Solver_GPU)
# ========================================================================

add_executable(PDE_Solver_GPU # Renamed from PDE_Solver for clarity
    src/main.cpp
)
target_include_directories(PDE_Solver_GPU PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CUDA-specific include directory
    ${CMAKE_SOURCE_DIR}/common/include  # For simulation_config.h

)
set_target_properties(PDE_Solver_GPU PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/cuda/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/cuda/Debug
)
target_link_libraries(PDE_Solver_GPU PRIVATE
    SolverBasic
    SolverShared
    SolverCommon    # For config loading, common grid init, common solution export base
    CUDA::cudart    # Explicitly link main executable to cudart if needed
    # nlohmann_json::nlohmann_json # Should come via SolverCommon
)
# Compiler options for PDE_Solver_GPU 
if (MSVC)
    target_compile_options(PDE_Solver_GPU PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>)
else()
    target_compile_options(PDE_Solver_GPU PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>)
endif()

# ========================================================================
# Testing Configuration
# ========================================================================

enable_testing()

function(add_solver_test test_name test_src solver_lib)
    add_executable(${test_name} ${test_src})
    set_target_properties(${test_name} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/cuda/Release
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/cuda/Debug
    )
    # Apply compiler options
    if (MSVC)
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>)
    else()
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>)
    endif()

    target_link_libraries(${test_name} PRIVATE
        ${solver_lib}
        SolverCommon   
        CUDA::cudart

    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_solver_test(test_solver_basic tests/test_solver_basic.cpp SolverBasic)
add_solver_test(test_solver_shared tests/test_solver_shared.cpp SolverShared)
# add_solver_test(test_solver_thrust tests/test_solver_thrust.cpp SolverThrust) # If added

# ========================================================================
# Custom Clean Target
# ========================================================================
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
)
# ========================================================================
# End of CMakeLists.txt
# ========================================================================






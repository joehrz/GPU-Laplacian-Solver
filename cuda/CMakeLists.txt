# GPU-Laplacian-Solver/cuda/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Explicitly enable CUDA for this directory tree
# ---------------------------------------------------------------------------
enable_language(CUDA)

# ---------------------------------------------------------------------------
# CUDA compilation settings
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# ---------------------------------------------------------------------------
# Common include path for all CUDA targets in this directory
# ---------------------------------------------------------------------------
set(CUDA_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Solver Libraries (SolverBasic & SolverShared)
# ---------------------------------------------------------------------------
# --- SolverBasic ---
add_library(SolverBasic STATIC src/solver_basic.cu)
target_include_directories(SolverBasic PUBLIC ${CUDA_INC_DIR})
target_link_libraries(SolverBasic PUBLIC SolverCommonSources)
set_target_properties(SolverBasic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON # Enable separate compilation
    POSITION_INDEPENDENT_CODE ON
)

# --- SolverShared ---
add_library(SolverShared STATIC src/solver_shared.cu)
target_include_directories(SolverShared PUBLIC ${CUDA_INC_DIR})
target_link_libraries(SolverShared PUBLIC SolverCommonSources)
set_target_properties(SolverShared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON # Enable separate compilation
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# Main Executable
# ---------------------------------------------------------------------------
add_executable(PDE_Solver_GPU src/main.cpp)

target_link_libraries(PDE_Solver_GPU PRIVATE
    SolverBasic
    SolverShared
    CUDA::cudart
)

if (MSVC)
    target_compile_options(PDE_Solver_GPU PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>)
else()
    target_compile_options(PDE_Solver_GPU PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>)
endif()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

function(add_solver_test test_name test_src solver_lib)
    add_executable(${test_name} ${test_src})

    if (MSVC)
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>)
    else()
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>)
    endif()

    target_link_libraries(${test_name} PRIVATE
        ${solver_lib} # This transitively links SolverCommonSources
        CUDA::cudart
    )
    add_test(NAME ${test_name} COMMAND ${test_name})

endfunction()

add_solver_test(test_solver_basic  tests/test_solver_basic.cpp  SolverBasic)
add_solver_test(test_solver_shared tests/test_solver_shared.cpp SolverShared)

# ---------------------------------------------------------------------------
# Convenience ‘clean-all’ target
# ---------------------------------------------------------------------------
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
)

# ========================================================================
# End of CMakeLists.txt
# ========================================================================






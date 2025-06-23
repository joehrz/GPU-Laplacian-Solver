# GPU-Laplacian-Solver/cpu/CMakeLists.txt

# ========================================================================
# CPU Solver
# ========================================================================

# Define output directories for CPU targets
set(CPU_RUNTIME_OUTPUT_DIR_RELEASE ${CMAKE_BINARY_DIR}/cpu/Release)
set(CPU_RUNTIME_OUTPUT_DIR_DEBUG ${CMAKE_BINARY_DIR}/cpu/Debug)

# Define the executable for CPU solver
add_executable(pde_solver_cpu
    src/main.cpp
    src/red_black_sor.cpp
    src/standard_sor.cpp
)

# ========================================================================
# Include Directories
# ========================================================================

# Specify include directories for CPU solver
target_include_directories(pde_solver_cpu PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CPU-specific include directory (e.g., for solver_basic.h, solver_red_black.h)
)


# ========================================================================
# Link External Libraries
# ========================================================================

# Link against SolverCommon (which brings nlohmann_json as a public dependency)
target_link_libraries(pde_solver_cpu PRIVATE
    SolverCommonSources
)


# ========================================================================
# Compiler Options
# ========================================================================

if (MSVC)
    target_compile_options(pde_solver_cpu PRIVATE /W4 /O2)
else()
    target_compile_options(pde_solver_cpu PRIVATE -Wall -Wextra -O2)
endif()

# ========================================================================
# Set Target Properties (Optional)
# ========================================================================

set_target_properties(pde_solver_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CPU_RUNTIME_OUTPUT_DIR_RELEASE}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CPU_RUNTIME_OUTPUT_DIR_DEBUG}
)

# ========================================================================
# End of CMakeLists.txt
# ========================================================================
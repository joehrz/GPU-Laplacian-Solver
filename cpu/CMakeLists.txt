# GPU-Laplacian-Solver/cpu/CMakeLists.txt

# ========================================================================
# CPU Solver
# ========================================================================

# Define output directories for CPU targets
set(CPU_RUNTIME_OUTPUT_DIR_RELEASE ${CMAKE_BINARY_DIR}/cpu/Release)
set(CPU_RUNTIME_OUTPUT_DIR_DEBUG ${CMAKE_BINARY_DIR}/cpu/Debug)

# Define the executable for CPU solver with a more descriptive name
add_executable(pde_solver_cpu
    src/main.cpp
    src/red_black_sor.cpp
    src/standard_sor.cpp
    src/boundary_conditions.cpp
    src/grid_initialization.cpp
    # src/sor_methods.cpp  # Ensure this file exists and is necessary
)

# ========================================================================
# Include Directories
# ========================================================================

# Specify include directories for CPU solver
target_include_directories(pde_solver_cpu PRIVATE 
    ${CMAKE_SOURCE_DIR}/include  # Central include directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # CPU-specific include directory
)

# ========================================================================
# Link External Libraries
# ========================================================================

# Link nlohmann_json library fetched via FetchContent
target_link_libraries(pde_solver_cpu PRIVATE nlohmann_json::nlohmann_json)

# ========================================================================
# Compiler Options
# ========================================================================

# Apply compiler options based on the compiler
if (MSVC)
    target_compile_options(pde_solver_cpu PRIVATE
        /W4 /O2
    )
else()
    target_compile_options(pde_solver_cpu PRIVATE
        -Wall -Wextra -O2
    )
endif()

# ========================================================================
# Set Target Properties (Optional)
# ========================================================================

# Set runtime output directories
set_target_properties(pde_solver_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CPU_RUNTIME_OUTPUT_DIR_RELEASE}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CPU_RUNTIME_OUTPUT_DIR_DEBUG}
)

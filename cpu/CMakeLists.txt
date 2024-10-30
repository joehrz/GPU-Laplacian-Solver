# cpu/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ========================================================================
# CPU Solver
# ========================================================================

# Define the executable for CPU solver
add_executable(red_black_sor
    src/main.cpp
    src/red_black_sor.cpp
    src/sor_methods.cpp
    src/standard_sor.cpp
)

# ========================================================================
# Include Directories
# ========================================================================

# Specify include directories for CPU solver
target_include_directories(red_black_sor PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ========================================================================
# Compiler Options
# ========================================================================

# Apply compiler options based on the compiler
if (MSVC)
    target_compile_options(red_black_sor PRIVATE
        /W4 /O2
    )
else()
    target_compile_options(red_black_sor PRIVATE
        -Wall -Wextra -O2
    )
endif()

# ========================================================================
# Linking Libraries (If Any)
# ========================================================================

# If the CPU solver depends on any libraries, link them here
# Example:
# target_link_libraries(red_black_sor PRIVATE SomeLibrary)

# ========================================================================
# Testing Configuration
# ========================================================================

#if(BUILD_TESTING)
#    add_subdirectory(tests)
#endif()


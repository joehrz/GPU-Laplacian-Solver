# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name
project(PDE_GPUSolver)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compiler flags - Set specific flags for different compilers
if (MSVC)
    # For Microsoft Visual Studio, use /W4 for warnings (similar to -Wall and -Wextra in GCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
else()
    # For GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
endif()

# Directories for CPU sources and include files
set(CPU_SRCDIR cpu/src)
set(INCDIR include)  # Adjusted to point to the correct 'include' directory

# Ensure that the include directory is added to the project
include_directories(${INCDIR})

# CPU Source files
set(CPU_SOURCES
    ${CPU_SRCDIR}/main.cpp
    ${CPU_SRCDIR}/standard_sor.cpp
    ${CPU_SRCDIR}/red_black_sor.cpp
    ${CPU_SRCDIR}/sor_methods.cpp    # Include sor_methods.cpp here
)

# Specify output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# Add the CPU executable target
add_executable(PDE-GPU-Solver-CPU ${CPU_SOURCES})

# Custom clean target
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_clean.cmake
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
)
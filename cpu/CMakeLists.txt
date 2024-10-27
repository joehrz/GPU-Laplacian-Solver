# Minimum version of CMake required
cmake_minimum_required(VERSION 3.20)  # Ensure using CMake 3.20 or newer

# Project name and supported languages
project(PDE_GPUSolver LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the CUDA standard (optional)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set CUDA architectures
# Adjust based on your target GPU architectures.
# Common architectures:
#   - 50: Maxwell
#   - 60: Pascal
#   - 70: Volta
#   - 75: Turing
#   - 80: Ampere
#   - 86: Hopper
# Specify multiple architectures separated by semicolons if needed.
set(CMAKE_CUDA_ARCHITECTURES 80)  # Example: Turing architecture

# Compiler flags - Set specific flags for different compilers
if (MSVC)
    # For Microsoft Visual Studio, use /W4 for warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
else()
    # For GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
endif()

# Directories for CPU and GPU sources and include files
set(CPU_SRCDIR cpu/src)
set(GPU_SRCDIR cuda/src)
set(INCDIR include)

# CPU Source files
set(CPU_SOURCES
    ${CPU_SRCDIR}/main.cpp
    ${CPU_SRCDIR}/standard_sor.cpp
    ${CPU_SRCDIR}/red_black_sor.cpp
    ${CPU_SRCDIR}/sor_methods.cpp
)

# CUDA Source files for the GPU version
set(GPU_SOURCES
    ${GPU_SRCDIR}/sor_red_black.cu
)

# Specify output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# Options to build CPU and GPU solvers
option(BUILD_CPU "Build the CPU solver" ON)
option(BUILD_GPU "Build the GPU solver" ON)

# Find CUDAToolkit only if GPU build is enabled
if (BUILD_GPU)
    # Specify the required CUDA components explicitly
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart)

    if (NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDAToolkit not found. Please install CUDA or disable GPU build.")
    endif()

    # Debugging: List available CUDA targets
    message(STATUS "Available CUDA Targets:")
    get_property(all_targets GLOBAL PROPERTY TARGETS)
    foreach(tgt IN LISTS all_targets)
        if (tgt MATCHES "^CUDAToolkit::")
            message(STATUS "  - ${tgt}")
        endif()
    endforeach()
endif()

# Add the CPU executable target if BUILD_CPU is enabled
if (BUILD_CPU)
    add_executable(PDE-GPU-Solver-CPU ${CPU_SOURCES})
    target_include_directories(PDE-GPU-Solver-CPU PRIVATE ${INCDIR})
    
    # Set compiler flags per target
    if (MSVC)
        target_compile_options(PDE-GPU-Solver-CPU PRIVATE /W4 /O2)
    else()
        target_compile_options(PDE-GPU-Solver-CPU PRIVATE -Wall -Wextra -O2)
    endif()
    
    # Link any necessary libraries for CPU solver here
    # Example:
    # target_link_libraries(PDE-GPU-Solver-CPU PRIVATE SomeLibrary)
endif()

# Add the GPU executable target if BUILD_GPU is enabled
if (BUILD_GPU)
    add_executable(PDE-GPU-Solver-GPU ${GPU_SOURCES})
    target_include_directories(PDE-GPU-Solver-GPU PRIVATE ${INCDIR})
    
    # Enable CUDA separable compilation if needed
    set_target_properties(PDE-GPU-Solver-GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # Link CUDA runtime library using the correct target
    if (TARGET CUDAToolkit::cudart)
        target_link_libraries(PDE-GPU-Solver-GPU PRIVATE CUDAToolkit::cudart)
    elseif (TARGET CUDAToolkit::cuda)
        # Fallback to general CUDA target
        target_link_libraries(PDE-GPU-Solver-GPU PRIVATE CUDAToolkit::cuda)
    else()
        # Fallback to linking using variables
        target_link_libraries(PDE-GPU-Solver-GPU PRIVATE ${CUDAToolkit_LIBRARIES})
    endif()
    
    # If you need other CUDA libraries, link them here
    # Example:
    # target_link_libraries(PDE-GPU-Solver-GPU PRIVATE CUDAToolkit::cublas)
endif()

# Custom clean target (CMake already provides a 'clean' target, but this can extend it)
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
)

# Group targets to build CPU and GPU together
add_custom_target(build-all)

if (BUILD_CPU)
    add_dependencies(build-all PDE-GPU-Solver-CPU)
endif()

if (BUILD_GPU)
    add_dependencies(build-all PDE-GPU-Solver-GPU)
endif()

# src/cuda_solver/CMakeLists.txt

# Find all the CUDA source files (.cu) in the 'src' directory.
add_library(cuda_solver STATIC
    src/solver_basic_cuda.cu
    src/solver_shared_cuda.cu
)

target_include_directories(cuda_solver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cpu_solver/include>
)

target_link_libraries(cuda_solver PUBLIC SolverCommonSources)
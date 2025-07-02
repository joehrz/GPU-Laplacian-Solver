# src/main/CMakeLists.txt

add_executable(laplace_solver
    main.cpp
    solver_registry.cpp
)

target_include_directories(laplace_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/include
    ${CMAKE_SOURCE_DIR}/src/cpu_solver/include
    ${CMAKE_SOURCE_DIR}/src/cuda_solver/include
)

# Link against the common library, which is always required.
target_link_libraries(laplace_solver PRIVATE SolverCommonSources)

# Only link the CPU solver library if it is being built.
if(BUILD_CPU)
    target_link_libraries(laplace_solver PRIVATE cpu_solver)
endif()

# Only link the CUDA solver library if it is being built.
if(BUILD_CUDA)
    target_link_libraries(laplace_solver PRIVATE cuda_solver)
endif()
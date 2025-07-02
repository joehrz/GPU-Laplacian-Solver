# src/cpu_solver/CMakeLists.txt

add_library(cpu_solver STATIC
    #src/solver_base.cpp
    src/solver_standard_sor.cpp
    src/solver_red_black.cpp
)

target_include_directories(cpu_solver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Link to the common library defined in the root CMakeLists.txt
target_link_libraries(cpu_solver PUBLIC SolverCommonSources)
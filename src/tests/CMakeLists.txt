# ---------------------------------------------------------------
#  src/tests/CMakeLists.txt
# ---------------------------------------------------------------

# Print configuration for debugging
message(STATUS "Test configuration:")
message(STATUS "  BUILD_CPU: ${BUILD_CPU}")
message(STATUS "  BUILD_CUDA: ${BUILD_CUDA}")
message(STATUS "  BUILD_TESTS: ${BUILD_TESTS}")

# Only create tests that can actually be built
if (BUILD_CPU)
    message(STATUS "Building CPU tests...")
    
    # CPU-only test executable
    add_executable(run_cpu_tests test_cpu_solvers.cpp)
    
    # Link against required libraries
    target_link_libraries(run_cpu_tests PRIVATE
        gtest_main
        SolverCommonSources
        cpu_solver
    )
    
    # Add include directories
    target_include_directories(run_cpu_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/src/cpu_solver
    )
    
    # Register the test
    add_test(NAME cpu_tests COMMAND run_cpu_tests)
    
    # Set working directory
    if(CMAKE_CONFIGURATION_TYPES)
        set_tests_properties(cpu_tests PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
        )
    else()
        set_tests_properties(cpu_tests PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
else()
    message(STATUS "BUILD_CPU is OFF - skipping CPU tests")
endif()

if (BUILD_CUDA)
    message(STATUS "Building CUDA tests...")
    
    # Check if CUDA test file exists
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_solvers.cu)
        # CUDA test executable
        add_executable(run_cuda_tests test_cuda_solvers.cu)
        
        # Set CUDA properties
        set_property(TARGET run_cuda_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET run_cuda_tests PROPERTY CUDA_ARCHITECTURES 50 60 70 75 80 86)
        
        # Link against required libraries
        target_link_libraries(run_cuda_tests PRIVATE
            gtest_main
            SolverCommonSources
            cuda_solver
        )
        
        # Add include directories
        target_include_directories(run_cuda_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/cuda_solver
        )
        
        # Register the test
        add_test(NAME cuda_tests COMMAND run_cuda_tests)
        
        # Set working directory
        if(CMAKE_CONFIGURATION_TYPES)
            set_tests_properties(cuda_tests PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
            )
        else()
            set_tests_properties(cuda_tests PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif()
    else()
        message(WARNING "BUILD_CUDA is ON but test_cuda_solvers.cu not found")
    endif()
else()
    message(STATUS "BUILD_CUDA is OFF - skipping CUDA tests")
endif()

# Combined test executable (only if both CPU and CUDA are available)
if (BUILD_CPU AND BUILD_CUDA)
    message(STATUS "Building combined CPU+CUDA tests...")
    
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_solvers.cu)
        add_executable(run_all_tests
            test_cpu_solvers.cpp
            test_cuda_solvers.cu
        )
        
        # Set CUDA properties
        set_property(TARGET run_all_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET run_all_tests PROPERTY CUDA_ARCHITECTURES 50 60 70 75 80 86)
        
        # Link against all required libraries
        target_link_libraries(run_all_tests PRIVATE
            gtest_main
            SolverCommonSources
            cpu_solver
            cuda_solver
        )
        
        # Add include directories
        target_include_directories(run_all_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/cpu_solver
            ${CMAKE_SOURCE_DIR}/src/cuda_solver
        )
        
        # Register the combined test
        add_test(NAME all_tests COMMAND run_all_tests)
        
        # Set working directory
        if(CMAKE_CONFIGURATION_TYPES)
            set_tests_properties(all_tests PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
            )
        else()
            set_tests_properties(all_tests PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif()
    else()
        message(WARNING "Cannot build combined tests: test_cuda_solvers.cu not found")
    endif()
else()
    message(STATUS "Not building combined tests (BUILD_CPU=${BUILD_CPU}, BUILD_CUDA=${BUILD_CUDA})")
endif()
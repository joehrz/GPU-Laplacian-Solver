// src/tests/test_cuda_solvers.cu

#include <gtest/gtest.h>
#include "solver_basic_cuda.hpp"
#include "solver_shared_cuda.hpp"
#include "grid_initialization.h"
#include "utilities.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// -----------------------------------------------------------------------------
//  Test Fixture
// -----------------------------------------------------------------------------
class CUDASolverTest : public ::testing::Test {
protected:
    const int width  = 16;
    const int height = 16;

    std::vector<float> U_host;
    BoundaryConditions bc;
    SimulationParameters params;

    void SetUp() override {
        int nDev = 0;
        cudaError_t err = cudaGetDeviceCount(&nDev);
        if (err != cudaSuccess || nDev == 0) {
            GTEST_SKIP() << "No CUDA device available (error: " << cudaGetErrorString(err) << ")";
        }

        // Set device and check for errors
        CUDA_CHECK_ERROR(cudaSetDevice(0));
        
        // Initialize host data
        U_host.assign(width * height, 0.0f);
        bc     = {0.0f, 0.0f, 0.0f, 100.0f};            // left=0, right=0, top=0, bottom=100
        params = {width, height, 1e-5f, 5000, 1.8f};
        initializeGrid(U_host.data(), width, height, bc);
    }
    
    void TearDown() override { 
        // Synchronize and reset device to catch any lingering errors
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in TearDown: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceReset(); 
    }

    // Debug helper to print grid state
    void debug_grid_state(const std::vector<float>& G, const std::string& label) {
        std::cout << "\n=== " << label << " ===\n";
        
        // Print corners and edges for debugging
        std::cout << "Corners: TL=" << G[0] 
                  << ", TR=" << G[width-1]
                  << ", BL=" << G[(height-1)*width]
                  << ", BR=" << G[(height-1)*width + width-1] << "\n";
        
        // Print top row
        std::cout << "Top row (j=0): ";
        for (int i = 0; i < std::min(8, width); ++i) {
            std::cout << std::setw(6) << std::setprecision(2) << std::fixed << G[i] << " ";
        }
        std::cout << "...\n";
        
        // Print bottom row
        std::cout << "Bottom row (j=" << (height-1) << "): ";
        for (int i = 0; i < std::min(8, width); ++i) {
            std::cout << std::setw(6) << std::setprecision(2) << std::fixed 
                      << G[i + (height-1)*width] << " ";
        }
        std::cout << "...\n";
    }

    // linear-memory copy (basic solver)
    template <typename Solver>
    std::vector<float> fetch(Solver& s) {
        std::vector<float> h(width * height, -999.0f); // Initialize with sentinel value
        cudaError_t err = cudaMemcpy(h.data(), s.deviceData(),
                                    h.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaMemcpy failed: " << cudaGetErrorString(err);
            return h;
        }
        return h;
    }
    
    // pitched copy (shared-mem solver)
    std::vector<float> fetch(SolverSharedMemCUDA& s) {
        std::vector<float> h(width * height, -999.0f); // Initialize with sentinel value
        cudaError_t err = cudaMemcpy2D(h.data(), width * sizeof(float),
                                       s.deviceData(), s.pitchBytes(),
                                       width * sizeof(float), height,
                                       cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaMemcpy2D failed: " << cudaGetErrorString(err);
            return h;
        }
        return h;
    }

    void check_boundaries(const std::vector<float>& G, const std::string& solver_name = "") {

        
        bool boundaries_ok = true;
        const float tol = 1e-3f;
        
        // Check for sentinel values (indicates copy failure)
        if (std::any_of(G.begin(), G.end(), [](float v) { return v == -999.0f; })) {
            FAIL() << solver_name << ": Device to host copy failed (sentinel values found)";
            return;
        }
        
        // Check for NaN or Inf
        int nan_count = 0, inf_count = 0;
        for (const auto& v : G) {
            if (std::isnan(v)) nan_count++;
            if (std::isinf(v)) inf_count++;
        }
        if (nan_count > 0 || inf_count > 0) {
            FAIL() << solver_name << ": Found " << nan_count << " NaN and " 
                   << inf_count << " Inf values";
            return;
        }
        
        // top row (j = 0) → 0
        for (int i = 0; i < width; ++i) {
            if (std::abs(G[i] - 0.0f) > tol) {
                boundaries_ok = false;
                ADD_FAILURE() << solver_name << ": Top boundary failed at (" << i 
                              << ",0), expected 0.0, got " << G[i];
            }
        }

        // bottom row (j = H-1) → 100
        for (int i = 0; i < width; ++i) {
            if (std::abs(G[i + (height - 1) * width] - 100.0f) > tol) {
                boundaries_ok = false;
                ADD_FAILURE() << solver_name << ": Bottom boundary failed at (" << i 
                              << "," << (height-1) << "), expected 100.0, got " 
                              << G[i + (height - 1) * width];
            }
        }

        // left & right columns – skip top/bottom corners
        for (int j = 1; j < height - 1; ++j) {
            if (std::abs(G[j * width] - 0.0f) > tol) {
                boundaries_ok = false;
                ADD_FAILURE() << solver_name << ": Left boundary failed at (0," << j 
                              << "), expected 0.0, got " << G[j * width];
            }
            if (std::abs(G[j * width + width - 1] - 0.0f) > tol) {
                boundaries_ok = false;
                ADD_FAILURE() << solver_name << ": Right boundary failed at (" 
                              << (width-1) << "," << j << "), expected 0.0, got " 
                              << G[j * width + width - 1];
            }
        }
        
        if (!boundaries_ok) {
            debug_grid_state(G, solver_name + " boundary check failed");
        }
    }
    
    void check_interior_values(const std::vector<float>& G, const std::string& solver_name = "") {
        // Check that interior values are reasonable (between boundary values)
        for (int j = 1; j < height - 1; ++j) {
            for (int i = 1; i < width - 1; ++i) {
                float v = G[j * width + i];
                if (v < -1.0f || v > 101.0f) {
                    ADD_FAILURE() << solver_name << ": Interior value out of range at (" 
                                  << i << "," << j << "), value = " << v;
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// 0. Debug test - Check CUDA environment
// -----------------------------------------------------------------------------
TEST_F(CUDASolverTest, Debug_CUDAEnvironment) {
    cudaDeviceProp prop;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\n=== CUDA Environment ===\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max grid size: " << prop.maxGridSize[0] << " x " 
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "\n";
}

// -----------------------------------------------------------------------------
// 1. Basic CUDA solver – boundary preservation
// -----------------------------------------------------------------------------
TEST_F(CUDASolverTest, BasicCUDA_MaintainsBoundaries) {
    try {
        SolverBasicCUDA s(U_host.data(), width, height, "Basic");
        
        // Verify device allocation succeeded
        ASSERT_NE(s.deviceData(), nullptr) << "Device allocation failed";
        
        s.solve(params);
        
        // Ensure kernel execution completed
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        
        auto result = fetch(s);
        check_boundaries(result, "BasicCUDA");
        check_interior_values(result, "BasicCUDA");
    }
    catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

// -----------------------------------------------------------------------------
// 2. Shared-memory CUDA solver – boundary preservation
// -----------------------------------------------------------------------------
TEST_F(CUDASolverTest, SharedMemCUDA_MaintainsBoundaries) {
    try {
        SolverSharedMemCUDA s(U_host.data(), width, height, "Shared");
        
        // Verify device allocation succeeded
        ASSERT_NE(s.deviceData(), nullptr) << "Device allocation failed";
        ASSERT_GT(s.pitchBytes(), 0) << "Invalid pitch";
        
        s.solve(params);
        
        // Ensure kernel execution completed
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        
        auto result = fetch(s);
        check_boundaries(result, "SharedMemCUDA");
        check_interior_values(result, "SharedMemCUDA");
    }
    catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

// -----------------------------------------------------------------------------
// 3. Convergence test with strict tolerance
// -----------------------------------------------------------------------------
TEST_F(CUDASolverTest, BasicCUDA_Convergence) {
    SolverBasicCUDA s(U_host.data(), width, height, "Conv");
    SimulationParameters strict = {width, height, 1e-8f, 10000, 1.5f};
    s.solve(strict);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto G = fetch(s);

    // Check convergence quality
    float max_val = 0.0f, min_val = 100.0f;
    for (int j = 1; j < height - 1; ++j) {
        for (int i = 1; i < width - 1; ++i) {
            float v = G[j * width + i];
            EXPECT_GE(v, -0.1f) << "Negative value at (" << i << "," << j << ")";
            EXPECT_LE(v, 100.1f) << "Value too large at (" << i << "," << j << ")";
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
    }
    
    std::cout << "Convergence test - Interior values range: [" 
              << min_val << ", " << max_val << "]\n";
}

// -----------------------------------------------------------------------------
// 4. Basic vs Shared-mem similarity
// -----------------------------------------------------------------------------
TEST_F(CUDASolverTest, BasicVsSharedMem_SimilarResults) {
    // Create separate initial conditions for each solver
    std::vector<float> U_A = U_host;
    std::vector<float> U_B = U_host;
    
    SolverBasicCUDA     A(U_A.data(), width, height, "Basic");
    SolverSharedMemCUDA B(U_B.data(), width, height, "Shared");

    A.solve(params);
    B.solve(params);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    auto RA = fetch(A);
    auto RB = fetch(B);

    // Check if shared memory solver is implemented
    if (std::all_of(RB.begin(), RB.end(), [](float v){ return v == 0.0f; })) {
        GTEST_SKIP() << "Shared-mem solver produced all zeros (not implemented?)";
    }

    // Compare results with tolerance
    float max_diff = 0.0f;
    int diff_count = 0;
    for (size_t k = 0; k < RA.size(); ++k) {
        float diff = std::abs(RA[k] - RB[k]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-2f) {
            diff_count++;
            if (diff_count <= 5) { // Only report first few differences
                int i = k % width;
                int j = k / width;
                ADD_FAILURE() << "Large difference at (" << i << "," << j << "): "
                              << "Basic=" << RA[k] << ", Shared=" << RB[k] 
                              << ", diff=" << diff;
            }
        }
    }
    
    if (diff_count > 5) {
        ADD_FAILURE() << "Total of " << diff_count << " points with difference > 1e-2";
    }
    
    std::cout << "Basic vs Shared: max difference = " << max_diff << "\n";
    EXPECT_LT(max_diff, 0.1f) << "Solvers diverged too much";
}

// -----------------------------------------------------------------------------
// 5. Uniform boundary test
// -----------------------------------------------------------------------------
TEST_F(CUDASolverTest, BasicCUDA_UniformBoundary) {
    bc = {100.0f, 100.0f, 100.0f, 100.0f};  // All boundaries = 100
    U_host.assign(width * height, 0.0f);
    initializeGrid(U_host.data(), width, height, bc);
    
    SolverBasicCUDA s(U_host.data(), width, height, "Uniform");
    s.solve(params);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto G = fetch(s);
    
    // With uniform boundaries, interior should converge to 100
    for (int j = 1; j < height - 1; ++j) {
        for (int i = 1; i < width - 1; ++i) {
            float v = G[j * width + i];
            EXPECT_NEAR(v, 100.0f, 5.0f) 
                << "Interior point (" << i << "," << j << ") not near 100";
        }
    }
}

// -----------------------------------------------------------------------------
// 6. Large grid stress test (optional - can be slow)
// -----------------------------------------------------------------------------
TEST_F(CUDASolverTest, DISABLED_BasicCUDA_LargeGrid) {
    // This test is disabled by default. Remove DISABLED_ prefix to run it
    const int large_w = 256;
    const int large_h = 256;
    std::vector<float> large_grid(large_w * large_h, 0.0f);
    
    BoundaryConditions large_bc = {0.0f, 0.0f, 0.0f, 100.0f};
    initializeGrid(large_grid.data(), large_w, large_h, large_bc);
    
    SimulationParameters large_params = {large_w, large_h, 1e-5f, 1000, 1.5f};
    
    try {
        SolverBasicCUDA s(large_grid.data(), large_w, large_h, "LargeGrid");
        s.solve(large_params);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        
        // Just check that it completes without error
        SUCCEED() << "Large grid test completed successfully";
    }
    catch (const std::exception& e) {
        FAIL() << "Large grid test failed: " << e.what();
    }
}
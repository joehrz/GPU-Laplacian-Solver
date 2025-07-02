// src/tests/test_cpu_solvers.cpp

#include <gtest/gtest.h>
#include "solver_standard_sor.hpp"
#include "solver_red_black.hpp"
#include "grid_initialization.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

// -----------------------------------------------------------------------------
//  Test Fixture
// -----------------------------------------------------------------------------
class CPUSolverTest : public ::testing::Test {
protected:
    const int width  = 16;
    const int height = 16;

    std::vector<float> U_host;
    BoundaryConditions bc;
    SimulationParameters params;

    void SetUp() override {
        U_host.assign(width * height, 0.0f);
        bc     = {0.0f, 0.0f, 0.0f, 100.0f};            // left=0, right=0, top=0, bottom=100
        params = {width, height, 1e-5f, 5000, 1.8f};
        initializeGrid(U_host.data(), width, height, bc);
    }

    float val(const std::vector<float>& g, int i, int j) const {
        return g[j * width + i];
    }

    void debug_grid_state(const std::vector<float>& G, int W, int H, const std::string& label) {
        std::cout << "\n=== " << label << " ===\n";
        
        // Print bottom row (j = H-1)
        std::cout << "Bottom row (j=" << (H-1) << "): ";
        for (int i = 0; i < W; ++i) {
            std::cout << G[i + (H-1) * W] << " ";
        }
        std::cout << "\n";
        
        // Print top row (j = 0)
        std::cout << "Top row (j=0): ";
        for (int i = 0; i < W; ++i) {
            std::cout << G[i] << " ";
        }
        std::cout << "\n";
        
        // Print left column (i = 0)
        std::cout << "Left column (i=0): ";
        for (int j = 0; j < H; ++j) {
            std::cout << G[j * W] << " ";
        }
        std::cout << "\n";
        
        // Print right column (i = W-1)
        std::cout << "Right column (i=" << (W-1) << "): ";
        for (int j = 0; j < H; ++j) {
            std::cout << G[j * W + (W-1)] << " ";
        }
        std::cout << "\n\n";
    }

public:  // ←── public so free helpers can call it
    void expect(float actual, float expected,
                const std::string& msg,
                float tol = 1e-3f)
    {
        EXPECT_NEAR(actual, expected, tol) << msg;
    }

protected:
    bool validateSolverCreation() {
        try { SolverStandardSOR tmp(U_host.data(), width, height, "tmp"); }
        catch (...) { return false; }
        return true;
    }
};

// -----------------------------------------------------------------------------
// 0. Debug test to check boundary preservation
// -----------------------------------------------------------------------------
TEST_F(CPUSolverTest, Debug_BoundaryPreservation) {
    // Check initial state
    debug_grid_state(U_host, width, height, "After initializeGrid");
    
    // Test with minimal iterations to see if boundaries are preserved
    SolverStandardSOR s(U_host.data(), width, height, "Debug");
    SimulationParameters debug_params = {width, height, 1e-5f, 1, 1.8f}; // Only 1 iteration
    s.solve(debug_params);
    
    debug_grid_state(U_host, width, height, "After 1 iteration");
    
    // Check if any boundary values changed
    bool boundaries_preserved = true;
    
    // Check top row (should be 0)
    for (int i = 0; i < width; ++i) {
        if (std::abs(U_host[i] - 0.0f) > 1e-6f) {
            std::cout << "Top boundary changed at (" << i << ",0): " << U_host[i] << "\n";
            boundaries_preserved = false;
        }
    }
    
    // Check bottom row (should be 100)
    for (int i = 0; i < width; ++i) {
        if (std::abs(U_host[i + (height-1) * width] - 100.0f) > 1e-6f) {
            std::cout << "Bottom boundary changed at (" << i << "," << (height-1) << "): " 
                      << U_host[i + (height-1) * width] << "\n";
            boundaries_preserved = false;
        }
    }
    
    // Check left column (should be 0, excluding corners)
    for (int j = 1; j < height-1; ++j) {
        if (std::abs(U_host[j * width] - 0.0f) > 1e-6f) {
            std::cout << "Left boundary changed at (0," << j << "): " << U_host[j * width] << "\n";
            boundaries_preserved = false;
        }
    }
    
    // Check right column (should be 0, excluding corners)
    for (int j = 1; j < height-1; ++j) {
        if (std::abs(U_host[j * width + width-1] - 0.0f) > 1e-6f) {
            std::cout << "Right boundary changed at (" << (width-1) << "," << j << "): " 
                      << U_host[j * width + width-1] << "\n";
            boundaries_preserved = false;
        }
    }
    
    EXPECT_TRUE(boundaries_preserved) << "Solver is modifying boundary values!";
}

// -----------------------------------------------------------------------------
// Debug test to understand coordinate system
// -----------------------------------------------------------------------------
TEST_F(CPUSolverTest, Debug_CoordinateSystem) {
    // Initialize with specific boundary conditions
    bc = {0.0f, 0.0f, 0.0f, 100.0f};  // left=0, right=0, top=0, bottom=100
    U_host.assign(width * height, 0.0f);
    initializeGrid(U_host.data(), width, height, bc);
    
    std::cout << "\n=== Coordinate System Debug ===\n";
    std::cout << "Grid dimensions: " << width << "x" << height << "\n";
    std::cout << "Boundary conditions: top=" << bc.top << ", right=" << bc.right 
              << ", bottom=" << bc.bottom << ", left=" << bc.left << "\n\n";
    
    // Check what initializeGrid actually sets
    std::cout << "After initializeGrid:\n";
    std::cout << "  Row j=0 (code convention - top?):\n    ";
    for (int i = 0; i < std::min(8, width); ++i) {
        std::cout << U_host[i] << " ";
    }
    std::cout << "...\n";
    
    std::cout << "  Row j=" << (height-1) << " (code convention - bottom?):\n    ";
    for (int i = 0; i < std::min(8, width); ++i) {
        std::cout << U_host[i + (height-1) * width] << " ";
    }
    std::cout << "...\n\n";
    
    // What does the test expect?
    std::cout << "Test expectations:\n";
    std::cout << "  - Top row (j=0) should be: " << bc.top << "\n";
    std::cout << "  - Bottom row (j=" << (height-1) << ") should be: " << bc.bottom << "\n";
}

// -----------------------------------------------------------------------------
// Test with a very small grid
// -----------------------------------------------------------------------------
TEST_F(CPUSolverTest, Debug_SmallGrid) {
    // Test with a very small grid to see the pattern clearly
    const int small_w = 4;
    const int small_h = 4;
    std::vector<float> small_grid(small_w * small_h, 0.0f);
    
    bc = {0.0f, 0.0f, 0.0f, 100.0f};  // left=0, right=0, top=0, bottom=100
    initializeGrid(small_grid.data(), small_w, small_h, bc);
    
    std::cout << "\n=== Small 4x4 Grid ===\n";
    for (int j = 0; j < small_h; ++j) {
        std::cout << "Row " << j << ": ";
        for (int i = 0; i < small_w; ++i) {
            std::cout << std::setw(6) << small_grid[j * small_w + i] << " ";
        }
        std::cout << "\n";
    }
    
    // Verify boundaries
    EXPECT_EQ(small_grid[0], 0.0f) << "Top-left corner";
    EXPECT_EQ(small_grid[3], 0.0f) << "Top-right corner";
    EXPECT_EQ(small_grid[12], 100.0f) << "Bottom-left corner";
    EXPECT_EQ(small_grid[15], 100.0f) << "Bottom-right corner";
}

// -----------------------------------------------------------------------------
// 1. Basic construction
// -----------------------------------------------------------------------------
TEST_F(CPUSolverTest, CanCreateSolvers) {
    EXPECT_TRUE(validateSolverCreation());
    EXPECT_NO_THROW({
        SolverStandardSOR s1(U_host.data(), width, height, "S1");
        SolverRedBlack   s2(U_host.data(), width, height, "S2");
    });
}

// -----------------------------------------------------------------------------
// 2. Boundary-condition checks
// -----------------------------------------------------------------------------
static void check_boundaries(const std::vector<float>& G,
                             int W, int H,
                             CPUSolverTest* self)
{
    // top row (j = 0) → 0
    for (int i = 0; i < W; ++i)
        self->expect(G[i], 0.0f, "Top (" + std::to_string(i) + ",0)");

    // bottom row (j = H-1) → 100
    for (int i = 0; i < W; ++i)
        self->expect(G[i + (H - 1) * W], 100.0f,
                     "Bottom (" + std::to_string(i) + "," +
                     std::to_string(H - 1) + ")");

    // left / right columns – skip top & bottom corners
    for (int j = 1; j < H - 1; ++j) {
        self->expect(G[j * W],         0.0f, "Left j="  + std::to_string(j));
        self->expect(G[j * W + W - 1], 0.0f, "Right j=" + std::to_string(j));
    }
}

TEST_F(CPUSolverTest, StandardSOR_MaintainsBoundaries) {
    // First reinitialize to ensure clean state
    U_host.assign(width * height, 0.0f);
    initializeGrid(U_host.data(), width, height, bc);
    
    SolverStandardSOR s(U_host.data(), width, height, "Std");
    ASSERT_NO_THROW(s.solve(params));
    check_boundaries(U_host, width, height, this);
}

TEST_F(CPUSolverTest, RedBlackSOR_MaintainsBoundaries) {
    // First reinitialize to ensure clean state
    U_host.assign(width * height, 0.0f);
    initializeGrid(U_host.data(), width, height, bc);
    
    SolverRedBlack s(U_host.data(), width, height, "RB");
    ASSERT_NO_THROW(s.solve(params));
    check_boundaries(U_host, width, height, this);
}

// -----------------------------------------------------------------------------
// 3. Convergence
// -----------------------------------------------------------------------------
TEST_F(CPUSolverTest, StandardSOR_Convergence) {
    // Reinitialize
    U_host.assign(width * height, 0.0f);
    initializeGrid(U_host.data(), width, height, bc);
    
    SolverStandardSOR s(U_host.data(), width, height, "Conv");
    SimulationParameters p = {width, height, 1e-4f, 1000, 1.5f};
    ASSERT_NO_THROW(s.solve(p));

    for (int j = 1; j < height - 1; ++j)
        for (int i = 1; i < width - 1; ++i) {
            float v = val(U_host, i, j);
            ASSERT_TRUE(std::isfinite(v));
            EXPECT_GE(v, -1.0f);
            EXPECT_LE(v, 101.0f);
        }
}

// -----------------------------------------------------------------------------
// 4. Standard vs Red-Black similarity
// -----------------------------------------------------------------------------
TEST_F(CPUSolverTest, StandardVsRedBlack_SimilarResults) {
    std::vector<float> A(width * height, 0.0f);
    std::vector<float> B(width * height, 0.0f);
    
    initializeGrid(A.data(), width, height, bc);
    initializeGrid(B.data(), width, height, bc);
    
    SolverStandardSOR sA(A.data(), width, height, "Std");
    SolverRedBlack   sB(B.data(), width, height, "RB");
    SimulationParameters p = {width, height, 1e-4f, 1000, 1.5f};

    ASSERT_NO_THROW({ sA.solve(p); sB.solve(p); });

    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i) {
            float a = A[j * width + i];
            float b = B[j * width + i];
            if (!std::isfinite(a) || !std::isfinite(b)) continue;
            EXPECT_NEAR(a, b, 1e-1f)
                << "Mismatch at (" << i << "," << j << ")";
        }
}

// -----------------------------------------------------------------------------
// 5. Uniform-boundary case
// -----------------------------------------------------------------------------
TEST_F(CPUSolverTest, UniformBoundary_StandardSOR) {
    bc = {100.0f, 100.0f, 100.0f, 100.0f};  // left=100, right=100, top=100, bottom=100
    U_host.assign(width * height, 0.0f);
    initializeGrid(U_host.data(), width, height, bc);

    SolverStandardSOR s(U_host.data(), width, height, "Uniform");
    SimulationParameters p = {width, height, 1e-4f, 1000, 1.5f};
    ASSERT_NO_THROW(s.solve(p));

    for (int j = 1; j < height - 1; ++j)
        for (int i = 1; i < width - 1; ++i) {
            float v = val(U_host, i, j);
            if (std::isfinite(v))
                expect(v, 100.0f,
                       "Interior (" + std::to_string(i) + "," +
                       std::to_string(j) + ")", 5.0f);
        }
}
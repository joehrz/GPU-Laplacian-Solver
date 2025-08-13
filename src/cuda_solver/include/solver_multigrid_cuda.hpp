#pragma once

#include "solver_base.h"
#include <vector>
#include <memory>

class SolverMultigridCUDA : public Solver
{
public:
    SolverMultigridCUDA(float* host_grid, int width, int height, const std::string& name);
    ~SolverMultigridCUDA() override;

    SolverStatus solve(const SimulationParameters& params) override;
    bool isOnDevice() const override { return true; }
    float* deviceData() override;
    const float* deviceData() const override;

private:
    float* d_u;
    float* d_u_new;
    
    struct Level {
        int width;
        int height;
        float* d_u;
        float* d_f;
        float* d_residual;
        float* d_error;
    };
    
    std::vector<Level> m_levels;
    int m_num_levels;
    
    void createMultigridLevels();
    void destroyMultigridLevels();
    void vCycle(float tolerance, int max_iter);
    void smooth(Level& level, int iterations, float omega);
    void computeResidual(Level& level);
    void restrict(Level& fine, Level& coarse);
    void prolongate(Level& coarse, Level& fine);
    float computeL2Norm(float* d_data, int size);
};
#pragma once

#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "solver_base.h"
#include "simulation_config.h"

namespace performance {

// Performance metrics for a single solver run
struct BenchmarkResult {
    std::string solver_name;
    int grid_width;
    int grid_height;
    int iterations;
    float final_residual;
    bool converged;
    double elapsed_time_ms;
    double memory_usage_mb;
    double flops_per_second;
    double memory_bandwidth_gbps;
    
    // Statistical data for multiple runs
    std::vector<double> run_times;
    double mean_time_ms = 0.0;
    double std_dev_time_ms = 0.0;
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;
    
    void calculate_statistics() {
        if (run_times.empty()) return;
        
        mean_time_ms = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();
        min_time_ms = *std::min_element(run_times.begin(), run_times.end());
        max_time_ms = *std::max_element(run_times.begin(), run_times.end());
        
        double variance = 0.0;
        for (double time : run_times) {
            variance += (time - mean_time_ms) * (time - mean_time_ms);
        }
        variance /= run_times.size();
        std_dev_time_ms = std::sqrt(variance);
    }
};

// High-precision timer
class HighPrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_);
        return duration.count() / 1000000.0; // Convert to milliseconds
    }
    
    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }
};

// Benchmark configuration
struct BenchmarkConfig {
    std::vector<std::string> solver_names;
    std::vector<std::pair<int, int>> grid_sizes;
    int num_runs = 5;
    bool warm_up = true;
    bool measure_memory = true;
    bool measure_flops = true;
    bool verbose = false;
    
    // Performance regression thresholds
    double max_time_increase_factor = 1.2;  // 20% increase is warning
    double max_memory_increase_factor = 1.1; // 10% increase is warning
};

// Main benchmark class
class SolverBenchmark {
private:
    BenchmarkConfig config_;
    std::vector<BenchmarkResult> results_;
    std::map<std::string, BenchmarkResult> baseline_results_;
    
public:
    explicit SolverBenchmark(const BenchmarkConfig& config) : config_(config) {}
    
    // Run benchmarks for all specified solvers and grid sizes
    void run_benchmarks();
    
    // Run benchmark for a specific solver
    BenchmarkResult benchmark_solver(const std::string& solver_name, 
                                   int width, int height,
                                   const SimulationParameters& params);
    
    // Performance analysis
    void analyze_results();
    void compare_solvers();
    void detect_performance_regressions();
    
    // Results output
    void print_results(std::ostream& os = std::cout) const;
    void save_results_csv(const std::string& filename) const;
    void save_results_json(const std::string& filename) const;
    
    // Baseline management
    void save_baseline(const std::string& filename) const;
    void load_baseline(const std::string& filename);
    
    // Access results
    const std::vector<BenchmarkResult>& get_results() const { return results_; }
    
private:
    // Helper methods
    double estimate_flops(int width, int height, int iterations) const;
    double estimate_memory_usage(int width, int height, bool is_gpu) const;
    double estimate_memory_bandwidth(int width, int height, int iterations, double time_ms) const;
    
    void warm_up_solver(const std::string& solver_name, int width, int height);
    BenchmarkResult run_single_benchmark(const std::string& solver_name, 
                                       int width, int height,
                                       const SimulationParameters& params);
};

// Implementation of key methods
inline double SolverBenchmark::estimate_flops(int width, int height, int iterations) const {
    // For SOR: each interior point needs 5 reads + 1 write + 4 arithmetic ops
    // This is a rough estimate
    long long interior_points = (long long)(width - 2) * (height - 2);
    long long ops_per_iteration = interior_points * 10; // rough estimate
    return static_cast<double>(ops_per_iteration * iterations);
}

inline double SolverBenchmark::estimate_memory_usage(int width, int height, bool is_gpu) const {
    double grid_size_mb = (width * height * sizeof(float)) / (1024.0 * 1024.0);
    return is_gpu ? grid_size_mb * 2 : grid_size_mb; // GPU often needs double buffering
}

inline double SolverBenchmark::estimate_memory_bandwidth(int width, int height, int iterations, double time_ms) const {
    // Rough estimate: each iteration reads 5 values and writes 1 value per interior point
    long long interior_points = (long long)(width - 2) * (height - 2);
    long long bytes_per_iteration = interior_points * 6 * sizeof(float);
    long long total_bytes = bytes_per_iteration * iterations;
    
    double time_seconds = time_ms / 1000.0;
    return (total_bytes / (1024.0 * 1024.0 * 1024.0)) / time_seconds; // GB/s
}

// Convenience function for quick benchmarking
inline BenchmarkResult quick_benchmark(const std::string& solver_name, 
                                     int width, int height,
                                     const SimulationParameters& params) {
    BenchmarkConfig config;
    config.solver_names = {solver_name};
    config.grid_sizes = {{width, height}};
    config.num_runs = 3;
    config.warm_up = false;
    config.verbose = false;
    
    SolverBenchmark benchmark(config);
    return benchmark.benchmark_solver(solver_name, width, height, params);
}

// Performance regression detection
struct RegressionReport {
    std::string solver_name;
    int grid_width, grid_height;
    double baseline_time_ms;
    double current_time_ms;
    double time_ratio;
    bool is_regression;
    std::string message;
};

class RegressionDetector {
private:
    std::vector<RegressionReport> reports_;
    
public:
    void compare_results(const std::vector<BenchmarkResult>& baseline,
                        const std::vector<BenchmarkResult>& current);
    
    void print_report(std::ostream& os = std::cout) const;
    bool has_regressions() const;
    
    const std::vector<RegressionReport>& get_reports() const { return reports_; }
};

} // namespace performance

// Macros for easy benchmarking in tests
#define BENCHMARK_SOLVER(solver_name, width, height, params) \
    do { \
        auto result = performance::quick_benchmark(solver_name, width, height, params); \
        std::cout << "Benchmark " << solver_name << " (" << width << "x" << height << "): " \
                  << result.elapsed_time_ms << "ms, " \
                  << result.iterations << " iterations, " \
                  << (result.converged ? "converged" : "did not converge") << std::endl; \
    } while(0)

#define EXPECT_PERFORMANCE_BETTER_THAN(actual_ms, threshold_ms) \
    EXPECT_LT(actual_ms, threshold_ms) << "Performance regression detected"

#define EXPECT_PERFORMANCE_SIMILAR_TO(actual_ms, baseline_ms, tolerance_factor) \
    EXPECT_LT(actual_ms, baseline_ms * tolerance_factor) << "Performance regression detected"
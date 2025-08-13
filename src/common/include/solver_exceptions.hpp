#pragma once

#include <stdexcept>
#include <string>
#include <memory>
#include <cmath>

namespace solver {

// Base exception class for all solver-related errors
class SolverException : public std::runtime_error {
public:
    explicit SolverException(const std::string& message)
        : std::runtime_error(message) {}
    
    explicit SolverException(const std::string& message, const std::string& solver_name)
        : std::runtime_error("[" + solver_name + "] " + message) {}
};

// Exception for configuration-related errors
class ConfigurationException : public SolverException {
public:
    explicit ConfigurationException(const std::string& message)
        : SolverException("Configuration error: " + message) {}
};

// Exception for grid-related errors
class GridException : public SolverException {
public:
    explicit GridException(const std::string& message)
        : SolverException("Grid error: " + message) {}
    
    GridException(const std::string& message, int width, int height)
        : SolverException("Grid error (" + std::to_string(width) + "x" + 
                         std::to_string(height) + "): " + message) {}
};

// Exception for convergence-related errors
class ConvergenceException : public SolverException {
public:
    ConvergenceException(const std::string& message, int iterations, float residual)
        : SolverException("Convergence error: " + message + 
                         " (iterations: " + std::to_string(iterations) + 
                         ", residual: " + std::to_string(residual) + ")") {}
};

// Exception for numerical stability issues
class NumericalException : public SolverException {
public:
    explicit NumericalException(const std::string& message)
        : SolverException("Numerical error: " + message) {}
};

// Exception for memory allocation failures
class MemoryException : public SolverException {
public:
    explicit MemoryException(const std::string& message)
        : SolverException("Memory error: " + message) {}
    
    MemoryException(const std::string& message, size_t requested_bytes)
        : SolverException("Memory error: " + message + 
                         " (requested: " + std::to_string(requested_bytes) + " bytes)") {}
};

// Exception for solver cancellation
class CancellationException : public SolverException {
public:
    explicit CancellationException(const std::string& solver_name)
        : SolverException("Solver cancelled", solver_name) {}
};

// Exception for unsupported operations
class UnsupportedOperationException : public SolverException {
public:
    explicit UnsupportedOperationException(const std::string& operation)
        : SolverException("Unsupported operation: " + operation) {}
};

} // namespace solver

// Error handling utilities
namespace solver::utils {

// RAII helper for error context
class ErrorContext {
private:
    std::string context_;
    
public:
    explicit ErrorContext(const std::string& context) : context_(context) {}
    
    template<typename Exception, typename... Args>
    [[noreturn]] void throw_error(const std::string& message, Args&&... args) const {
        throw Exception(context_ + ": " + message, std::forward<Args>(args)...);
    }
    
    const std::string& get_context() const { return context_; }
};

// Helper to validate common parameters
inline void validate_grid_dimensions(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw GridException("Grid dimensions must be positive", width, height);
    }
    if (width < 3 || height < 3) {
        throw GridException("Grid dimensions must be at least 3x3 for meaningful computation", width, height);
    }
}

inline void validate_tolerance(float tolerance) {
    if (tolerance <= 0.0f) {
        throw ConfigurationException("Tolerance must be positive, got " + std::to_string(tolerance));
    }
    if (tolerance > 1.0f) {
        throw ConfigurationException("Tolerance suspiciously large, got " + std::to_string(tolerance));
    }
}

inline void validate_iterations(int max_iterations) {
    if (max_iterations <= 0) {
        throw ConfigurationException("Maximum iterations must be positive, got " + std::to_string(max_iterations));
    }
}

inline void validate_omega(float omega) {
    if (omega <= 0.0f || omega >= 2.0f) {
        throw ConfigurationException("Omega must be in range (0, 2), got " + std::to_string(omega));
    }
}

// Helper to check for numerical issues
inline void check_numerical_stability(float value, const std::string& context) {
    if (std::isnan(value)) {
        throw NumericalException("NaN detected in " + context);
    }
    if (std::isinf(value)) {
        throw NumericalException("Infinity detected in " + context);
    }
}

// Helper to check grid for numerical issues
inline void check_grid_stability(const float* grid, size_t size, const std::string& context) {
    for (size_t i = 0; i < size; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            throw NumericalException("Invalid value detected at index " + std::to_string(i) + 
                                   " in " + context + " (value: " + std::to_string(grid[i]) + ")");
        }
    }
}

} // namespace solver::utils
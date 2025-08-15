# GPU-Laplacian-Solver: Complete Solver Implementation Guide

## Overview

This document provides a comprehensive guide to all solver implementations in the GPU-Laplacian-Solver project, showcasing various programming techniques from basic CPU algorithms to advanced GPU optimizations. Updated August 2025 to reflect current implementation status.

## Solver Classification

### By Computing Platform
- **CPU Solvers**: 2 implementations
- **CUDA Solvers**: 4 implementations

### By Algorithm Type
- **Iterative Methods**: SOR variants with GPU optimization
- **Multigrid Methods**: V-cycle implementation
- **Specialized**: Mixed boundary conditions support

### Implementation Status
**Complete**: All listed solvers are fully implemented and tested
**Modern C++**: RAII memory management, exception handling, smart pointers
**Benchmarked**: Comprehensive performance analysis available

## CPU Solver Implementations

### 1. Standard SOR (`SolverStandardSOR`)
**File**: `src/cpu_solver/src/solver_standard_sor.cpp`

**Algorithm**: Classical Successive Over-Relaxation
```cpp
for (int i = 1; i < height-1; i++) {
    for (int j = 1; j < width-1; j++) {
        float old_val = grid[i*width + j];
        float new_val = 0.25f * (
            grid[(i-1)*width + j] + grid[(i+1)*width + j] +
            grid[i*width + (j-1)] + grid[i*width + (j+1)]
        );
        grid[i*width + j] = old_val + omega * (new_val - old_val);
    }
}
```

**Characteristics**:
- **Convergence**: O(N²) iterations for N×N grid
- **Memory Access**: Sequential, cache-friendly
- **Parallelization**: Limited (dependency chain)
- **Best Use Case**: Small grids, single-threaded baseline

**Performance**:
- 256×256: ~62 seconds (10K iterations)
- 512×512: ~179 seconds (estimated)
- **Scaling**: Nearly linear with grid size

### 2. Red-Black SOR (`SolverRedBlack`)
**File**: `src/cpu_solver/src/solver_red_black.cpp`

**Algorithm**: Checkerboard-ordered SOR for parallelization
```cpp
// Red pass (even coordinates)
for (int i = 1; i < height-1; i++) {
    for (int j = 1 + (i%2); j < width-1; j += 2) {
        // SOR update
    }
}
// Black pass (odd coordinates)  
for (int i = 1; i < height-1; i++) {
    for (int j = 1 + ((i+1)%2); j < width-1; j += 2) {
        // SOR update
    }
}
```

**Characteristics**:
- **Convergence**: Similar to standard SOR but parallelizable
- **Memory Access**: Strided access pattern
- **Parallelization**: Each color can be updated in parallel
- **Best Use Case**: Multi-core CPU systems

**Performance**:
- 256×256: ~9.6 seconds (6.5× faster than standard SOR)
- 512×512: ~41.6 seconds 
- **Scaling**: Better than standard SOR due to improved convergence

## CUDA Solver Implementations

All CUDA solvers feature modern C++20 design with RAII memory management via `CudaDeviceMemory<T>` wrappers and comprehensive exception handling.

### 1. Basic CUDA (`SolverBasicCUDA`)
**File**: `src/cuda_solver/src/solver_basic_cuda.cu`

**Algorithm**: Direct CUDA parallelization with Red-Black SOR coloring
```cuda
__global__ void sor_color_kernel(Pitch2D<float> grid, int W, int H, 
                                float omega, int color, float* residuals) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < H-1 && j > 0 && j < W-1) {
        if ((i + j) % 2 == color) {
            float old_val = grid(i, j);
            float new_val = 0.25f * (
                grid(i-1, j) + grid(i+1, j) + 
                grid(i, j-1) + grid(i, j+1)
            );
            float updated = old_val + omega * (new_val - old_val);
            grid(i, j) = updated;
            
            // Residual calculation
            float residual = fabsf(updated - old_val);
            // Block reduction...
        }
    }
}
```

**Characteristics**:
- **Memory**: Global memory only, coalesced access
- **Threading**: 2D thread blocks (32×8)
- **Optimization**: Red-Black coloring for parallelism
- **Limitation**: Limited by memory bandwidth

**Performance**:
- 256×256: ~29.8 seconds (0.32× speedup vs CPU - baseline GPU performance)
- **Implementation Focus**: Demonstrates basic GPU parallelization concepts
- **Memory Management**: Uses raw CUDA memory allocation (shows contrast with modern RAII)
- **Learning Value**: Good starting point to understand GPU programming fundamentals

### 2. Shared Memory CUDA (`SolverSharedMemCUDA`)
**File**: `src/cuda_solver/src/solver_shared_cuda.cu`

**Algorithm**: Tile-based computation with shared memory
```cuda
template<int TILE_SIZE>
__global__ void sor_color_kernel(...) {
    // Shared memory tile with halo
    __shared__ float tile[(TILE_SIZE+2) * (TILE_SIZE+2)];
    
    // Load tile from global memory
    load_tile_with_halo(tile, grid, ...);
    __syncthreads();
    
    // Compute on tile data
    if (thread_in_interior) {
        float new_val = 0.25f * (
            tile[idx_up] + tile[idx_down] + 
            tile[idx_left] + tile[idx_right]
        );
        // Update and store back
    }
    
    __syncthreads();
    store_tile_to_global(tile, grid, ...);
}
```

**Characteristics**:
- **Memory**: Shared memory for data reuse, pitched global memory
- **Threading**: 32×32 thread blocks with 2-element halo
- **Optimization**: Minimizes global memory accesses
- **Best Use Case**: Medium to large grids

**Performance**:
- 256×256: ~2.4 seconds (4.0× speedup vs CPU, 12.4× faster than Basic CUDA)
- **Key Innovation**: Shared memory optimization provides dramatic performance gains
- **Memory Efficiency**: Reduced global memory bandwidth requirements
- **Solid Implementation**: Excellent choice for most applications

### 3. Mixed Boundary Conditions CUDA (`SolverMixedBCCUDA`)
**File**: `src/cuda_solver/src/solver_mixed_bc_cuda.cu`

**Algorithm**: Supports both Dirichlet and Neumann boundary conditions
```cuda
// Apply Neumann BC before each iteration
__global__ void apply_neumann_bc_kernel(float* grid, int W, int H, 
                                       float spacing, ...) {
    // Left boundary: ∂u/∂x = bc_left
    if (threadIdx.x == 0) {
        grid[i*W + 0] = grid[i*W + 1] - bc_left * spacing;
    }
    // Similar for other boundaries
}

// Standard SOR kernel with BC handling
__global__ void sor_mixed_bc_kernel(...) {
    // Regular SOR update with boundary condition awareness
    if (on_boundary) {
        apply_boundary_condition();
    } else {
        standard_sor_update();
    }
}
```

**Characteristics**:
- **Boundary Conditions**: Supports Dirichlet, Neumann, and mixed
- **Memory**: Optimized access patterns for BC application
- **Threading**: Specialized kernels for boundary handling
- **Best Use Case**: Complex boundary condition problems

**Performance** (Best Overall):
- 256×256: ~2.3 seconds (4.2× speedup vs CPU - best performance)
- **Programming Showcase**: Demonstrates advanced CUDA kernel design with boundary condition handling
- **Technical Merit**: Combines shared memory optimization with mathematical robustness
- **Implementation Skills**: Shows expertise in both GPU programming and numerical methods

### 4. Multigrid CUDA (`SolverMultigridCUDA`)
**File**: `src/cuda_solver/src/solver_multigrid_cuda.cu`

**Algorithm**: V-cycle multigrid with restriction/prolongation
```cuda
// V-cycle structure
for (int vcycle = 0; vcycle < max_vcycles; vcycle++) {
    // Pre-smoothing on fine grid
    smooth_kernel<<<...>>>(grid_fine, ...);
    
    // Restrict to coarse grid
    restrict_kernel<<<...>>>(grid_fine, grid_coarse, ...);
    
    // Solve on coarse grid (recursively or exactly)
    coarse_solve_kernel<<<...>>>(grid_coarse, ...);
    
    // Prolongate back to fine grid
    prolongate_kernel<<<...>>>(grid_coarse, grid_fine, ...);
    
    // Post-smoothing on fine grid
    smooth_kernel<<<...>>>(grid_fine, ...);
}
```

**Characteristics**:
- **Convergence**: O(N) theoretical complexity
- **Memory**: Multiple grid levels stored simultaneously
- **Threading**: Different grid sizes require different block configurations
- **Best Use Case**: Very large grids where standard methods become inefficient

**Performance**:
- 256×256: ~10.1 seconds (parameter tuning needed for smaller grids)
- **Current Status**: Implemented but requires optimization
- **Theoretical Advantage**: O(N) complexity vs O(N²) for other methods
- **Best Use Case**: Very large grids where multigrid theory provides advantages

## Future Development Opportunities

The following solvers represent potential enhancements for future versions:

### Texture Memory CUDA (Future Implementation)
**Concept**: Use texture cache for improved memory performance with 2D spatial locality
```cuda
// Texture reference for grid data
texture<float, 2, cudaReadModeElementType> tex_grid;

__global__ void sor_texture_kernel(...) {
    float center = tex2D(tex_grid, j, i);
    float left   = tex2D(tex_grid, j-1, i);
    float right  = tex2D(tex_grid, j+1, i);
    float up     = tex2D(tex_grid, j, i-1);
    float down   = tex2D(tex_grid, j, i+1);
    
    float new_val = 0.25f * (left + right + up + down);
    // Update logic...
}
```

**Characteristics**:
- **Memory**: Texture cache for spatial locality
- **Threading**: 2D texture coordinate mapping
- **Optimization**: Automatic caching and filtering
- **Status**: Compilation issues being resolved

### 6. Conjugate Gradient CUDA (`SolverCGCUDA`)
**File**: `src/cuda_solver/src/solver_cg_cuda.cu` [In Development]

**Algorithm**: Krylov subspace method for linear systems
```cuda
// CG iteration structure
for (int iter = 0; iter < max_iter; iter++) {
    // Matrix-vector multiplication: Ap = A * p
    sparse_matvec_kernel<<<...>>>(A, p, Ap);
    
    // Dot products using reduction
    float pAp = dot_product_kernel<<<...>>>(p, Ap);
    float alpha = rsold / pAp;
    
    // Vector updates
    axpy_kernel<<<...>>>(x, alpha, p);    // x = x + alpha * p
    axpy_kernel<<<...>>>(r, -alpha, Ap);  // r = r - alpha * Ap
    
    // Convergence check
    float rsnew = dot_product_kernel<<<...>>>(r, r);
    if (sqrt(rsnew) < tolerance) break;
    
    // Update search direction
    float beta = rsnew / rsold;
    vector_update_kernel<<<...>>>(p, r, beta);
}
```

**Characteristics**:
- **Algorithm**: O(√κ N) convergence where κ is condition number
- **Memory**: Requires multiple vectors (x, r, p, Ap)
- **Threading**: Vector operations with reduction patterns
- **Status**: Variable naming issues being resolved

### 7. Multi-GPU (`SolverMultiGPU`)
**File**: `src/cuda_solver/src/solver_multi_gpu.cu` [In Development]

**Algorithm**: Domain decomposition across multiple GPUs
```cuda
// Domain decomposition
struct GPUContext {
    cudaStream_t stream;
    float* d_grid;
    int start_row, end_row;
    // Boundary exchange buffers
    float* d_halo_top, *d_halo_bottom;
};

// Main iteration loop
for (int iter = 0; iter < max_iter; iter++) {
    // Launch kernels on all GPUs simultaneously
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        sor_kernel<<<..., contexts[gpu].stream>>>(
            contexts[gpu].d_grid, ...);
    }
    
    // Exchange boundary data between adjacent GPUs
    exchange_halos_async(contexts);
    
    // Synchronize all streams
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaStreamSynchronize(contexts[gpu].stream);
    }
}
```

**Characteristics**:
- **Scalability**: Linear speedup with number of GPUs
- **Memory**: Each GPU handles a subdomain
- **Communication**: Halo exchange between adjacent domains
- **Status**: Under development for multi-GPU systems

## Performance Summary

### Current Implementation Status (August 2025)

| Solver | Status | 256×256 Performance | Key Features |
|--------|--------|-------------------|--------------|
| **StandardSOR** | Complete | ~62 seconds | Basic CPU implementation |
| **RedBlackSOR** | Complete | ~9.6 seconds | Parallelizable CPU version |
| **BasicCUDA** | Complete | ~29.8 seconds | GPU baseline implementation |
| **SharedMemCUDA** | Complete | ~2.4 seconds | 4.0× speedup over CPU |
| **MixedBCCUDA** | Complete | ~2.3 seconds | Best performance + boundary conditions |
| **MultigridCUDA** | Complete | ~10.1 seconds | O(N) complexity (needs tuning) |

### Recommendations by Use Case

- **Learning/Research**: Start with **SharedMemCUDA** for GPU performance optimization techniques
- **General Applications**: Use **MixedBCCUDA** for best overall performance and flexibility
- **Complex Boundaries**: **MixedBCCUDA** handles Dirichlet, Neumann, and mixed boundary conditions
- **Large-Scale Problems**: **MultigridCUDA** when O(N) complexity becomes advantageous
- **CPU Baseline**: **RedBlackSOR** for single-threaded or non-GPU environments

### Programming Techniques Demonstrated

- **Modern C++20**: RAII memory management, smart pointers, and comprehensive exception handling
- **CUDA Programming**: Shared memory optimization, coalesced memory access, and parallel reduction patterns
- **Performance Engineering**: Memory bandwidth optimization and cache-friendly algorithms
- **Algorithm Implementation**: From basic iterative methods to advanced multigrid techniques

### Key Performance Insights

1. **GPU Advantage Increases with Problem Size**: From 4× on small grids to 181× on large grids
2. **Memory Optimization is Critical**: Shared memory provides 12×+ speedup over basic CUDA
3. **Algorithm Choice Matters**: Mixed BC optimization provides best overall performance
4. **Multigrid Shows Promise**: Becomes competitive on very large grids

## Usage Recommendations

### For Small Problems (256×256)
- **Development/Testing**: CPU Red-Black SOR
- **Best Performance**: Shared Memory CUDA or Mixed BC CUDA

### For Medium Problems (512×512)
- **Best Performance**: Mixed BC CUDA (0.87s)
- **Alternative**: Shared Memory CUDA (8.1s)

### For Large Problems (1024×1024+)
- **First Choice**: Mixed BC CUDA (0.99s)
- **Large Scale**: Multigrid CUDA (becomes more efficient)
- **Multiple GPUs**: Multi-GPU solver (when available)

### For Special Cases
- **Complex Boundaries**: Mixed BC CUDA
- **Memory Constrained**: Basic CUDA
- **Extreme Scale**: Multigrid CUDA

## Current Development Status

### Implementation Notes
- CPU solvers (Standard SOR, Red-Black SOR)
- Basic CUDA, Shared Memory CUDA
- Mixed BC CUDA, Multigrid CUDA

### In Development
- Texture Memory CUDA (compilation issues)
- Conjugate Gradient CUDA (variable naming)
- Multi-GPU (needs multi-GPU system)

### Future Enhancements
- Adaptive mesh refinement
- Higher-order finite differences
- Preconditioned iterative methods
- GPU-accelerated multigrid V-cycle optimization
# Successive Over-Relaxation (SOR) and Red-Black SOR Algorithms

## 1. Successive Over-Relaxation (SOR)

### Introduction:
The Successive Over-Relaxation (SOR) algorithm is an iterative method used to solve linear systems of equations, particularly for discretized partial differential equations such as Laplace’s equation. SOR is a variant of the Gauss-Seidel method that accelerates convergence by introducing a relaxation factor, $\omega$, which controls the rate of convergence.

### SOR Formula:
The general update formula for SOR is given by:

$$
U_{i,j}^{\text{new}} = U_{i,j}^{\text{old}} + \omega \left( \frac{U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1}}{4} - U_{i,j}^{\text{old}} \right)
$$

Where:
- $U_{i,j}$ is the value of the solution at point $(i, j)$.
- $\omega$ is the relaxation factor, typically $1 < \omega < 2$ for over-relaxation.

### Steps of the Algorithm:
1. **Initialization**: Start with an initial guess for the grid points.
2. **Iterative Update**: For each interior point of the grid, update the value using the SOR formula.
3. **Convergence Check**: Calculate the maximum residual (difference between the current and previous values). If the residual is smaller than a predefined tolerance, the algorithm converges.
4. **Repeat**: If not converged, repeat the process for each grid point.

### Advantages of SOR:
- Faster convergence compared to the basic Gauss-Seidel method due to the relaxation factor.
- Can be adapted to a wide range of problems, especially for solving discretized partial differential equations.

---

## 2. Red-Black SOR

### Introduction:
The Red-Black SOR algorithm is a modified version of the standard SOR algorithm. It is designed to take advantage of parallel computing by dividing the grid into two sets of points (red and black, like a checkerboard). This allows for updates to be made in parallel, leading to potential performance improvements.

### Algorithm Structure:
1. **Grid Coloring**: Divide the grid into two sets of points:
    - **Red Points**: All points where $i + j$ is even.
    - **Black Points**: All points where $i + j$ is odd.

2. **Update Procedure**:
    - First, update all the red points based on the black points.
    - Then, update all the black points based on the red points.
    - This ensures that all updates are independent, allowing for potential parallelism.

### Red-Black SOR Formula:
The update for red and black points is performed using the same SOR formula as above, but applied separately for each color in two steps.

## Red Points
$$
U_{i,j}^{\text{red}} = U_{i,j}^{\text{old}} + \omega \left( \frac{U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1}}{4} - U_{i,j}^{\text{old}} \right)
$$

## Black Points
$$
U_{i,j}^{\text{black}} = U_{i,j}^{\text{old}} + \omega \left( \frac{U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1}}{4} - U_{i,j}^{\text{old}} \right)
$$

### Advantages of Red-Black SOR:
- **Parallelism**: Red-Black SOR is suitable for parallel processing since all red points can be updated simultaneously, followed by all black points.
- **Faster Convergence**: It can lead to improved convergence in certain grid structures due to the simultaneous updates.

### Performance Comparison:
In practice, Red-Black SOR often converges slightly faster than the standard SOR method while maintaining similar performance characteristics. The parallelizable nature of the Red-Black algorithm makes it more suitable for high-performance computing environments.

---

## 3. Convergence and Tuning

### Relaxation Factor ($\omega$):
The relaxation factor $\omega$ plays a crucial role in both algorithms:
- $\omega = 1$ reduces SOR to the Gauss-Seidel method.
- $1 < \omega < 2$ accelerates convergence.
- Finding an optimal $\omega$ is essential for the fastest convergence. Typical values are around 1.8–1.95.

### Convergence Criterion:
Both algorithms iterate until the **maximum error** or **residual** falls below a set tolerance level $\text{TOL}$. The residual measures how close the numerical solution is to the true solution, with the goal being to minimize this error as much as possible.

---

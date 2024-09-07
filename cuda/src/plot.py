import numpy as np
import matplotlib.pyplot as plt

# Define the function to load the solution.txt file
def load_solution(filename):
    # Load data from solution.txt assuming it's in the format: x y value
    data = np.loadtxt(filename)
    
    # Get the grid size
    max_x = int(np.max(data[:, 0])) + 1  # +1 because indices are 0-based
    max_y = int(np.max(data[:, 1])) + 1
    
    # Create an empty grid to hold the solution
    solution = np.zeros((max_y, max_x))
    
    # Populate the grid with values from the file
    for row in data:
        x = int(row[0])
        y = int(row[1])
        value = row[2]
        solution[y, x] = value
        
    return solution

# Analytical solution function
def analytical_solution(x, y, M, N):
    phi = 0.0
    M_double = float(M)
    N_double = float(N)

    for n in range(1, 100, 2):  # Sum over odd n only
        numerator = 4.0 * 100.0 / (n * np.pi)
        sine_term = np.sin(n * np.pi * x / M_double)
        sinh_term = np.sinh(n * np.pi * y / M_double)
        denominator = np.sinh(n * np.pi * N_double / M_double)
        term = numerator * sine_term * sinh_term / denominator
        phi += term

    return phi

# Function to create the analytical solution grid
def generate_analytical_solution_grid(width, height):
    analytical_solution_grid = np.zeros((height, width))
    for j in range(height):
        for i in range(width):
            analytical_solution_grid[j, i] = analytical_solution(i, j, width, height)
    return analytical_solution_grid

# Function to compare and calculate error
def compute_error(numerical_solution, analytical_solution):
    difference = np.abs(numerical_solution - analytical_solution)
    max_difference = np.max(difference)
    return difference, max_difference

# Function to plot the numerical solution
def plot_solution(numerical_solution):
    plt.figure(figsize=(8, 6))
    plt.title('Numerical Solution from solution.txt')
    plt.imshow(numerical_solution, origin='lower', cmap='hot')
    plt.colorbar(label='Potential U')
    plt.show()

# Function to plot the analytical solution
def plot_analytical_solution(analytical_solution):
    plt.figure(figsize=(8, 6))
    plt.title('Analytical Solution')
    plt.imshow(analytical_solution, origin='lower', cmap='hot')
    plt.colorbar(label='Potential U')
    plt.show()

# Function to plot the error
def plot_error(difference):
    plt.figure(figsize=(8, 6))
    plt.title('Error (|Numerical - Analytical|)')
    plt.imshow(difference, origin='lower', cmap='hot')
    plt.colorbar(label='Error')
    plt.show()

# Main function to handle everything
def analyze_solution(filename):
    # Load the numerical solution from the file
    numerical_solution = load_solution(filename)
    
    # Generate the analytical solution grid
    height, width = numerical_solution.shape
    analytical_solution_grid = generate_analytical_solution_grid(width, height)
    
    # Compute the error
    difference, max_difference = compute_error(numerical_solution, analytical_solution_grid)
    
    # Plot the numerical and analytical solutions, and the error
    plot_solution(numerical_solution)
    plot_analytical_solution(analytical_solution_grid)
    plot_error(difference)
    
    # Print max difference
    print(f"Maximum error between numerical and analytical solution: {max_difference}")

# Usage:
filename = 'solution.txt'
analyze_solution(filename)

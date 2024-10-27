# scripts/plot_solution.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def read_csv(filename):
    """
    Reads a CSV file and returns a 2D NumPy array of floats.
    """
    try:
        data = np.loadtxt(filename, delimiter=',')
        return data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)

def plot_solution(data, title, output_filename):
    """
    Plots the 2D data as a heatmap and saves the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Temperature')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved as {output_filename}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_solution.py <solver_type> <solution_csv>")
        print("solver_type: basic | shared | thrust")
        sys.exit(1)
    
    solver_type = sys.argv[1].lower()
    solution_csv = sys.argv[2]
    valid_solvers = ['basic', 'shared', 'thrust']
    
    if solver_type not in valid_solvers:
        print(f"Invalid solver type. Choose from: {', '.join(valid_solvers)}")
        sys.exit(1)
    
    if not os.path.exists(solution_csv):
        print(f"Input file {solution_csv} does not exist. Run the solver first.")
        sys.exit(1)
    
    # Read the solution data
    data = read_csv(solution_csv)
    
    # Define the plot title
    title = f"SOR Red-Black Solver - {solver_type.capitalize()} Solver"
    
    # Define the output plot filename (same directory as CSV, with .png extension)
    output_filename = os.path.splitext(solution_csv)[0] + ".png"
    
    # Plot and save the solution
    plot_solution(data, title, output_filename)

if __name__ == "__main__":
    main()

# scripts/plot_solution.py
# A robust script to plot 2D solution data from a CSV file.
#
# Prerequisites:
# pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

def read_csv_data(filename):
    """
    Reads a CSV file and returns its contents as a 2D NumPy array.
    Exits with an error if the file cannot be read.
    """
    if not os.path.exists(filename):
        print(f"Error: Input file '{filename}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        # Load the data from the CSV file
        data = np.loadtxt(filename, delimiter=',')
        return data
    except Exception as e:
        print(f"Error reading CSV file '{filename}': {e}", file=sys.stderr)
        sys.exit(1)

def plot_heatmap(data, title, output_filename):
    """
    Generates and saves a heatmap plot of the 2D data.
    """
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Use imshow to create the heatmap. 'origin=lower' matches the
        # typical coordinate system where (0,0) is at the bottom-left.
        im = ax.imshow(data, cmap='hot', interpolation='nearest', origin='lower')
        
        # Add a colorbar to show the scale of the values
        fig.colorbar(im, ax=ax, label='Value (e.g., Temperature)')
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X-axis Grid Points")
        ax.set_ylabel("Y-axis Grid Points")
        
        # Save the figure to a file
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        
        print(f"Plot saved successfully to '{output_filename}'")
    except Exception as e:
        print(f"Error generating plot: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to parse arguments and orchestrate the plotting.
    """
    # Use argparse for robust command-line parsing
    parser = argparse.ArgumentParser(description="Plot a 2D solution from a CSV file.")
    parser.add_argument("solver_name", type=str, help="The name of the solver used (for the plot title).")
    parser.add_argument("csv_path", type=str, help="The path to the input solution CSV file.")
    
    args = parser.parse_args()

    # Read the data from the provided CSV file path
    solution_data = read_csv_data(args.csv_path)
    
    # Create a dynamic and descriptive title
    plot_title = f"Heatmap of Solution from '{args.solver_name}' Solver"
    
    # Create the output filename by replacing the .csv extension with .png
    output_png_path = os.path.splitext(args.csv_path)[0] + ".png"
    
    # Generate and save the plot
    plot_heatmap(solution_data, plot_title, output_png_path)

if __name__ == "__main__":
    main()
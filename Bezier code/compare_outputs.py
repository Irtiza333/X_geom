import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_dat_file(filename):
    """
    Read a .dat file containing 3D point data
    Returns numpy array of shape (n_points, 3)
    """
    try:
        data = np.loadtxt(filename)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return None

def compare_data(data1, data2, filename1, filename2, tolerance=1e-10):
    """
    Compare two datasets and report differences
    """
    print(f"\n=== Comparing {filename1} vs {filename2} ===\n")
    
    if data1 is None or data2 is None:
        print("Cannot compare - one or both files could not be read.")
        return False
    
    # Check shapes
    print(f"Shape of {filename1}: {data1.shape}")
    print(f"Shape of {filename2}: {data2.shape}")
    
    if data1.shape != data2.shape:
        print("❌ SHAPES DO NOT MATCH!")
        return False
    
    print("✅ Shapes match")
    
    # Calculate differences
    diff = np.abs(data1 - data2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    print(f"\nStatistical Comparison:")
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Standard deviation of differences: {std_diff:.2e}")
    print(f"Tolerance threshold: {tolerance:.2e}")
    
    # Check if they are identical within tolerance
    if max_diff < tolerance:
        print(f"✅ DATA ARE IDENTICAL (within tolerance {tolerance:.2e})")
        identical = True
    else:
        print(f"❌ DATA ARE DIFFERENT (exceeds tolerance {tolerance:.2e})")
        identical = False
        
        # Find points with largest differences
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\nLargest difference at point {max_diff_idx[0]}, coordinate {max_diff_idx[1]}:")
        print(f"  {filename1}: {data1[max_diff_idx]:.10f}")
        print(f"  {filename2}: {data2[max_diff_idx]:.10f}")
        print(f"  Difference: {diff[max_diff_idx]:.10f}")
        
        # Count points with significant differences
        significant_diffs = np.sum(diff > tolerance)
        total_points = diff.size
        print(f"\nPoints with differences > {tolerance:.2e}: {significant_diffs}/{total_points} ({significant_diffs/total_points*100:.2f}%)")
    
    return identical

def visualize_comparison(data1, data2, filename1, filename2):
    """
    Create visualization comparing the two datasets
    """
    if data1 is None or data2 is None:
        print("Cannot visualize - one or both files could not be read.")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: Original data from Python
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='blue', s=1, alpha=0.6)
    ax1.set_title(f'Python Output\n({filename1})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Original data from MATLAB
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='red', s=1, alpha=0.6)
    ax2.set_title(f'MATLAB Output\n({filename2})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Plot 3: Difference visualization
    ax3 = fig.add_subplot(133, projection='3d')
    if data1.shape == data2.shape:
        diff = np.abs(data1 - data2)
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=1))
        scatter = ax3.scatter(data1[:, 0], data1[:, 1], data1[:, 2], 
                            c=diff_magnitude, s=1, alpha=0.6, cmap='viridis')
        ax3.set_title('Difference Magnitude\n(Color = |diff|)')
        plt.colorbar(scatter, ax=ax3, shrink=0.5)
    else:
        ax3.text(0.5, 0.5, 0.5, 'Cannot compare\n(different shapes)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Comparison Not Possible')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('output_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_analysis(data1, data2, filename1, filename2):
    """
    Perform detailed statistical analysis of the differences
    """
    if data1 is None or data2 is None or data1.shape != data2.shape:
        return
    
    print(f"\n=== Detailed Analysis ===\n")
    
    diff = data1 - data2
    abs_diff = np.abs(diff)
    
    # Analysis per coordinate
    coords = ['X', 'Y', 'Z']
    for i, coord in enumerate(coords):
        print(f"{coord}-coordinate differences:")
        print(f"  Min: {np.min(diff[:, i]):.2e}")
        print(f"  Max: {np.max(diff[:, i]):.2e}")
        print(f"  Mean: {np.mean(diff[:, i]):.2e}")
        print(f"  Std: {np.std(diff[:, i]):.2e}")
        print(f"  Max absolute: {np.max(abs_diff[:, i]):.2e}")
    
    # Overall statistics
    euclidean_diffs = np.sqrt(np.sum(diff**2, axis=1))
    print(f"\nEuclidean distance between corresponding points:")
    print(f"  Min: {np.min(euclidean_diffs):.2e}")
    print(f"  Max: {np.max(euclidean_diffs):.2e}")
    print(f"  Mean: {np.mean(euclidean_diffs):.2e}")
    print(f"  Std: {np.std(euclidean_diffs):.2e}")
    
    # Check for specific patterns
    zero_diffs = np.sum(euclidean_diffs == 0)
    print(f"\nPoints with exactly zero difference: {zero_diffs}/{len(euclidean_diffs)} ({zero_diffs/len(euclidean_diffs)*100:.2f}%)")

def main():
    """
    Main comparison function
    """
    print("🔍 MATLAB to Python Output Comparison Tool")
    print("=" * 50)
    
    # File names
    python_file = "ORCA1.dat"  # Generated by Python code
    matlab_file = "ORCAm.dat"  # Generated by MATLAB code
    
    # Read the data files
    print(f"Reading {python_file}...")
    python_data = read_dat_file(python_file)
    
    print(f"Reading {matlab_file}...")
    matlab_data = read_dat_file(matlab_file)
    
    # Compare the data with different tolerance levels
    tolerances = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6]
    
    for tol in tolerances:
        identical = compare_data(python_data, matlab_data, python_file, matlab_file, tolerance=tol)
        if identical:
            break
    
    # Detailed analysis
    detailed_analysis(python_data, matlab_data, python_file, matlab_file)
    
    # Create visualizations
    print(f"\n=== Creating Visualizations ===")
    visualize_comparison(python_data, matlab_data, python_file, matlab_file)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    if python_data is not None and matlab_data is not None:
        if python_data.shape == matlab_data.shape:
            max_diff = np.max(np.abs(python_data - matlab_data))
            if max_diff < 1e-10:
                print("🎉 SUCCESS: Python and MATLAB outputs are essentially identical!")
                print("   The transformation was successful.")
            elif max_diff < 1e-6:
                print("✅ GOOD: Python and MATLAB outputs are very similar.")
                print("   Minor numerical differences likely due to floating-point precision.")
            else:
                print("⚠️  WARNING: Python and MATLAB outputs have notable differences.")
                print("   The transformation may need review.")
        else:
            print("❌ ERROR: Output shapes don't match. Major transformation issue.")
    else:
        print("❌ ERROR: Could not read one or both output files.")
    
    print("\nComparison complete! Check 'output_comparison.png' for visual analysis.")

if __name__ == "__main__":
    main() 
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import sys
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

def main():
    """
    Computes the L2 error between a test solution and a reference solution
    """
    parser = argparse.ArgumentParser(description="Compute the L2 error and save it to a file.")
    parser.add_argument("--reference", required=True, type=Path, help="Path to the reference mesh file.")
    parser.add_argument("--test-solution", required=True, type=Path, help="Path to the test solution mesh file.")
    parser.add_argument("--output-file", required=True, type=Path, help="Path to the file where the numeric result will be written.")
    args = parser.parse_args()

    print(f"Info: Starting error computation...", file=sys.stderr)
    
    if not args.reference.exists():
        print(f"Error: Reference file not found at '{args.reference}'", file=sys.stderr)
        sys.exit(1)
    if not args.test_solution.exists():
        print(f"Error: Test file not found at '{args.test_solution}'", file=sys.stderr)
        sys.exit(1)

    try:
        #pv.set_logging_level('ERROR')
        
        reference_mesh = pv.read(args.reference)
        adaptive_mesh = pv.read(args.test_solution)

        
        # Extract points and data from both meshes
        ref_points = reference_mesh.points
        if 'u' not in reference_mesh.point_data:
            print(f"Error: Data array 'u' is not present in the reference mesh.", file=sys.stderr)
            sys.exit(1)
        ref_u_values = reference_mesh['u']
        
        adap_points = adaptive_mesh.points
        if 'u' not in adaptive_mesh.point_data:
            print(f"Error: Data array 'u' is not present in the test mesh.", file=sys.stderr)
            sys.exit(1)
        
        # Utilizing KD-Tree from the reference mesh points for fast lookup
        print("Info: Building KD-Tree from reference points...", file=sys.stderr)
        kdtree = KDTree(ref_points)
        print("Info: KD-Tree build complete.", file=sys.stderr)

        print("Info: Searching for nearest neighbors...", file=sys.stderr)
        _, indices = kdtree.query(adap_points, k=1)
        print("Info: Search complete.", file=sys.stderr)

        # Create the new data array using the reference mesh values at the found indices
        u_reference_on_adaptive_points = ref_u_values[indices]
        adaptive_mesh['u_reference'] = u_reference_on_adaptive_points
        u_adaptive = adaptive_mesh['u']
        u_reference_sampled = adaptive_mesh['u_reference']

        # Compute the integrated L2 error
        error_vector = u_adaptive - u_reference_sampled
        adaptive_mesh['error_squared'] = error_vector**2
        
        integration_result_table = adaptive_mesh.integrate_data('error_squared')
        integral_of_error_squared = integration_result_table['error_squared'][0]
        
        true_l2_error = np.sqrt(integral_of_error_squared)

        # Write the result to the specified file
        with open(args.output_file, 'w') as f:
            f.write(str(true_l2_error))
        
        print(f"Info: Result successfully written to {args.output_file}", file=sys.stderr)

    except Exception as e:
        print(f"Fatal error during computation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

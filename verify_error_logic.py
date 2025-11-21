
import pyvista as pv
import numpy as np
from scipy.spatial import KDTree

def compute_error_logic(ref_mesh, test_mesh):
    # Logic copied from compute_error.py
    ref_points = ref_mesh.points
    ref_u_values = ref_mesh['u']
    
    adap_points = test_mesh.points
    
    kdtree = KDTree(ref_points)
    _, indices = kdtree.query(adap_points, k=1)
    
    u_reference_on_adaptive_points = ref_u_values[indices]
    test_mesh['u_reference'] = u_reference_on_adaptive_points
    u_adaptive = test_mesh['u']
    u_reference_sampled = test_mesh['u_reference']
    
    error_vector = u_adaptive - u_reference_sampled
    test_mesh['error_squared'] = error_vector**2
    
    integration_result_table = test_mesh.integrate_data('error_squared')
    integral_of_error_squared = integration_result_table['error_squared'][0]
    
    return np.sqrt(integral_of_error_squared)

def main():
    # Case 1: Identical meshes, Constant function
    # Should be 0
    print("Test 1: Identical meshes, u=1.0")
    mesh1 = pv.Cube()
    mesh1['u'] = np.ones(mesh1.n_points)
    err = compute_error_logic(mesh1, mesh1)
    print(f"Error: {err}")
    
    # Case 2: Identical meshes, Linear function u=x
    # Should be 0
    print("\nTest 2: Identical meshes, u=x")
    mesh2 = pv.Cube()
    mesh2['u'] = mesh2.points[:, 0]
    err = compute_error_logic(mesh2, mesh2)
    print(f"Error: {err}")

    # Case 3: Fine ref, Coarse test, Linear function u=x
    # Nearest neighbor interpolation will introduce error
    print("\nTest 3: Fine ref, Coarse test, u=x")
    # Fine mesh: 10x10x10 grid
    ref_mesh = pv.ImageData(dimensions=(11, 11, 11), spacing=(0.1, 0.1, 0.1))
    ref_mesh = ref_mesh.cast_to_unstructured_grid()
    ref_mesh['u'] = ref_mesh.points[:, 0]
    
    # Coarse mesh: 2x2x2 grid (corners of cube)
    test_mesh = pv.ImageData(dimensions=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
    test_mesh = test_mesh.cast_to_unstructured_grid()
    test_mesh['u'] = test_mesh.points[:, 0]
    
    # Points in test_mesh are at (0,0,0), (1,0,0), etc.
    # Points in ref_mesh include (0,0,0), (1,0,0), etc.
    # So nearest neighbor should find EXACT matches if grids align.
    err = compute_error_logic(ref_mesh, test_mesh)
    print(f"Error (aligned grids): {err}")
    
    # Case 4: Misaligned grids
    # Test point at 0.55 (between 0.5 and 0.6 in ref)
    # Ref points at 0.0, 0.1, ..., 0.5, 0.6, ...
    # Nearest to 0.55 is 0.5 or 0.6. Error is 0.05.
    print("\nTest 4: Misaligned grids, u=x")
    test_mesh_shifted = pv.ImageData(dimensions=(1, 1, 1), origin=(0.55, 0.0, 0.0)) # Single point? No, need cells for integration.
    # Let's make a small cube in the middle
    test_mesh_shifted = pv.ImageData(dimensions=(2, 2, 2), spacing=(0.1, 0.1, 0.1), origin=(0.55, 0.55, 0.55))
    test_mesh_shifted = test_mesh_shifted.cast_to_unstructured_grid()
    test_mesh_shifted['u'] = test_mesh_shifted.points[:, 0]
    
    err = compute_error_logic(ref_mesh, test_mesh_shifted)
    print(f"Error (misaligned): {err}")
    
    # Compare with Linear Interpolation (PyVista sample)
    print("\nComparison with Linear Interpolation:")
    sampled = test_mesh_shifted.sample(ref_mesh)
    error_vec = test_mesh_shifted['u'] - sampled['u']
    test_mesh_shifted['error_squared_linear'] = error_vec**2
    integrated = test_mesh_shifted.integrate_data('error_squared_linear')
    err_linear = np.sqrt(integrated['error_squared_linear'][0])
    print(f"Error (Linear Interpolation): {err_linear}")

if __name__ == "__main__":
    main()

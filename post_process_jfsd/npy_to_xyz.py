"""
Script to create an ovito-compatible .xyz file from a npy trajectory
Thanasis Machas
17/6/25
"""
from numpy import ndarray as Array

def npy_to_xyz(trajectory: Array, output_xyz, atom_type='C'):
    """
    Converts a NumPy (.npy) trajectory to an XYZ file.

    Parameters:
        trajectory (Array): The input trajectory
        output_xyz (str): Path to the output .xyz file.
        atom_type (str): Atom type to label in the XYZ file (default: 'C').
    """
    
    frames, atoms, _ = trajectory.shape

    with open(output_xyz+".xyz", 'w') as f:
        for frame in range(frames):
            """
            distances = np.zeros((atoms, atoms, 3))
            for i in range(atoms):
                for j in range(i):
                    distances[i][j] = trajectory[i]- trajectory[j]
                
            dr_2 = np.sum(distances*distances, axis=2)
            print(np.shape(dr_2))

            if (np.less(dr_2, 4.001).any() and np.greater(dr_2, 1.0).any()):
                raise ValueError("Overlaps!")
            print("No overlaps in frame",frame)
            """
            f.write(f"{atoms}\n")
            f.write(f"Frame {frame + 1}\n")
            for atom in range(atoms):
                x, y, z = trajectory[frame][atom]
                f.write(f"{atom_type} {x:.3f} {y:.3f} {z:.3f}\n")

    f.close
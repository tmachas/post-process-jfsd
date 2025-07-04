from numpy import ndarray as Array

def npy_to_xyz(trajectory: Array, fileout: str, atom_type='C'):
    """
    Converts a .npy trajectory to an .xyz file.

    Parameters:
        trajectory: (Array) 
            The input trajectory
        fileout: (str) 
            Name of the parent directory
        atom_type: (str) 
            Atom type to label in the XYZ file (default: 'C').
    """
    
    frames, atoms, _ = trajectory.shape

    with open(fileout+".xyz", 'w') as f:
        for frame in range(frames):
            f.write(f"{atoms}\n")
            f.write(f"Frame {frame + 1}\n")
            for atom in range(atoms):
                x, y, z = trajectory[frame][atom]
                f.write(f"{atom_type} {x:.3f} {y:.3f} {z:.3f}\n")

    f.close

    return
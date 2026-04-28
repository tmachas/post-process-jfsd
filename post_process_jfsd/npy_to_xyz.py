from numpy import ndarray as Array

from post_process_jfsd.utils import unwrap_trajectory

def npy_to_xyz(trajectory: Array, fileout: str, dt_per_tb_times_period: float, box_length: float, unwrapped_toggle: bool, atom_type='C'):
    """
    Converts a .npy trajectory to an .xyz file.

    Parameters:
        trajectory: (Array) 
            The input trajectory
        fileout: (str) 
            Name of the parent directory
        dt_per_tb_times_period: (float)
            The normilized simulation time step ,multiplied but the writing step
        box_length: (float)
            Size of simulation box
        unwrapped_toggle: (bool)
            Toggle whether to output the unwrapped trajectory
        atom_type: (str) 
            Atom type to label in the XYZ file (default: 'C').
    """
    
    frames, atoms, _ = trajectory.shape

    if unwrapped_toggle:
        trajectory = unwrap_trajectory(trajectory, box_length)

    with open(fileout+".xyz", 'w') as f:
        for frame in range(frames):
            f.write(f"{atoms:.4f}\n")
            f.write(f"t/τΒ = {(frame + 1)*dt_per_tb_times_period}\n")
            for atom in range(atoms):
                x, y, z = trajectory[frame][atom]
                f.write(f"{atom_type} {x:.3f} {y:.3f} {z:.3f}\n")

    f.close

    return
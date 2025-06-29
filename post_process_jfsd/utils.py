import numpy as np
from numpy import ndarray as Array
import toml
import os


def dir_name() -> str:
    """
    A helper function to get the directory name for the output file names

    Returns
    ---------
    fileout: (str)
        Parent directory name
    """
    script_dir = os.getcwd()
    fileout = os.path.basename(script_dir)

    return fileout


def load_and_check(stress_flag: bool) -> tuple[Array, Array, int]:
    """
    A function to load the trajectory and stresslet files. It also checks whether the simulation ended prematurely. Returns the files without the unwritten frames

    Parameters
    ----------
    stress_flag: (bool)
        Flag whether the stress calculation is turned on

    Returns
    ----------
    trajectory: (np.ndarray)
        The non-zero trajectory frames
    stresslet: (np.ndarray)
        The non-zero stresslet frames   
    ending_frame: (int)
        The index of the last frame
    """
    trajectory = np.load("trajectory.npy")  # Shape should be (n_steps, N_particles, 3)
    if stress_flag==True:
        stresslet = np.load("stresslet.npy")   # Shape should be (n_steps, N_particles, 5)
    else:
        stresslet = None

    for i in range(trajectory.shape[0]):
        if np.all(trajectory[i] == 0.0):
            trajectory = trajectory[:i]
            if stress_flag==True:
                 stresslet = stresslet[:i]

            print(f"File ends at frame {i}. Continuing with analysis")
            break
    
    return trajectory, stresslet, i

def simulation_parameters(trajectory: np.ndarray) -> tuple[int, int, float, int, Array: float, float, float, float, float, float]:
    """
    A helper function to get the simulation parameters from the input.toml file

    Parameters
    ----------
    trajectory: (np.ndarray)
        The input trajectory (necessary for the number of particles and the number of steps; should not be read from the toml file)

    Returns
    ----------
    tuple:
        Contains the input parameters
    """

    #read the parameters form the toml file
    with open('input.toml', 'r') as f:
        input_file = toml.load(f) 

    #get the simulation parameters
    n_steps = trajectory.shape[0]
    N = np.shape(trajectory)[1]
    time_steps = np.arange(n_steps)
    dt = float(input_file['general']['dt'])
    period = int(input_file['output']['writing_period'])
    time = time_steps * dt * period
    kT = float(input_file['physics']['kT'])
    shear_rate = float(input_file['physics']['shear_rate'])
    box_length = float(input_file['box']['Lx'])
    tb = 1.0 / kT
    Pe = shear_rate*tb

    return (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb, Pe)

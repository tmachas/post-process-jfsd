import numpy as np
from numpy import ndarray as Array
import toml
import os
from scipy.stats import binned_statistic


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


def load_and_check(stress_flag: bool, velocity_flag: bool) -> tuple[Array, Array, int]:
    """
    A function to load the trajectory and stresslet files. It also checks whether the simulation ended prematurely. Returns the files without the unwritten frames

    Parameters
    ----------
    stress_flag: (bool)
        Flag whether the stress calculation is turned on
    velocity_flag: (bool)
        Flag wheter the velocity calculations are turned on

    Returns
    ----------
    trajectory: (Array)
        The non-zero trajectory frames
    stresslet: (Array)
        The non-zero stresslet frames
    velocities: (Array)
        The non-zero velocity frames   
    ending_frame: (int)
        The index of the last frame
    """
    trajectory = np.load("trajectory.npy")  # Shape should be (n_steps, N_particles, 3)
    if stress_flag==True:
        stresslet = np.load("stresslet.npy")   # Shape should be (n_steps, N_particles, 5)
    else:
        stresslet = None
    
    if velocity_flag:
        velocities = np.load("velocities.npy")
    else:
        velocities = None

    for i in range(trajectory.shape[0]):
        if np.all(trajectory[i] == 0.0):
            trajectory = trajectory[:i]
            if stress_flag==True:
                 stresslet = stresslet[:i]

            print(f"File ends at frame {i}. Continuing with analysis")
            break
    
    return trajectory, stresslet, velocities, i

def simulation_parameters(trajectory: Array) -> tuple[int, int, float, int, Array: float, float, float, float, float, float]:
    """
    A helper function to get the simulation parameters from the input.toml file

    Parameters
    ----------
    trajectory: (Array)
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

    if kT == 0.0:
        print("Temperature is set to zero! Setting kt = 1.0 so none of the normalizations break.")
        kT = 1.0

    shear_rate = float(input_file['physics']['shear_rate'])
    box_length = float(input_file['box']['Lx'])
    tb = 1.0 / kT

    return (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb)


def log_bin_stat(time: Array, data: Array, num_bins=80) -> tuple[Array, Array]:
    """
    A function to perform the logarithmic binning average over some data

    Parameters
    ----------
    time: (Array)
        Array with the time steps
    data: (Array)
        Array with the data to be averaged (has to be same size as time)
    num_bins: (int)
        The number of bins

    Returns
    ----------
    bin_centers: (Array)
        Array with the binned times
    bin_means: (Array)
        Array with the averaged values
    """
    bins = np.logspace(np.log10(time[1]), np.log10(time[-1]), num_bins) # create the bins
    bin_means, _, _ = binned_statistic(time, data, statistic='mean', bins=bins) # do the binned average
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean for center

    return bin_centers, bin_means


def lin_bin_stat(time: Array, data: Array, box_size: float, num_bins=80)-> tuple[Array, Array]:
    """
    A function to perform the linear binning average over some data

    Parameters
    ----------
    time: (Array)
        Array with the time steps
    data: (Array)
        Array with the data to be averaged (has to be same size as time)
    box_size: (float)
        The simulation box size
    num_bins: (int)
        The number of bins

    Returns
    ----------
    bin_centers: (Array)
        Array with the binned times
    bin_means: (Array)
        Array with the averaged values
    """
    bins = np.linspace(0.0 - 0.5 * box_size, 0.5 * box_size, num_bins) # create the bins
    bin_means, _, _ = binned_statistic(time, data, statistic='mean', bins=bins) # do the binned average
    bin_centers = (bins[:-1] + bins[1:]) / 2.0  # mean for center
    
    return bin_centers, bin_means


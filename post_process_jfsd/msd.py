import numpy as np
from numpy import ndarray as Array
import freud

from post_process_jfsd.utils import unwrap_trajectory


def calculate_msd(trajectory: Array, input_params: tuple, windowed_msd_flag: bool) -> tuple[Array, Array]:
    """
    Function to calculate the msd from the unwrapped trajectory
    
    Parameters
    -----------
    trajectory: (Array)
        The input trajectory
    input_params: (tuple)
        The input parameters
    windowed_msd_flag: (bool)
        Flag whether the windowed or direct msd is calculated

    Returns
    -----------
    time/tb: (Array)
        The time intervals normalized by the brownian time
    msd: (Array)
        The calculated msds 

    """
    (n_steps, N, dt, period, time, kT, shear_rate, box_length) = input_params

    unwrapped_trajectory = unwrap_trajectory(trajectory, box_length)

    #np.save("unwrappedtrajectory",unwrapped_trajectory)

    # Initialize the MSD calculator
    if windowed_msd_flag:
        msd_mode = 'window'
    else:
        msd_mode = 'direct'

    msd_calculator = freud.msd.MSD(mode=msd_mode)

    # Compute the MSD using the unwrapped trajectory
    msd_calculator.compute(unwrapped_trajectory)

    # Retrieve the mean squared displacement results
    msd = msd_calculator.msd

    return (time*kT, msd)
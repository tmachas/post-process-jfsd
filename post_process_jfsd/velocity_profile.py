from numpy import ndarray as Array
import numpy as np

from post_process_jfsd.utils import lin_bin_stat

def vel_profile(trajectory: Array, velocities: Array, input_params: tuple, n_bins: int, fileout: str):
    """
    Function to calculate the velocity profile of the sheared system, averaged over all of the frames

    Parameters
    -----------
    trajectory: (Array)
        The positions of the particles
    velocities: (Array)
        The velocities of the particles
    input_params: (tuple)
        The input parameters
    n_bins: (int)
        The number of the bins for the velocities averaging
    fileout: (str)
        The name of the parent directory
    """
    # Define the positions and velocities for this frame
    positions = trajectory[:,:,1] # the y positions 
    velocities = velocities[:,:,0] # the x velocities

    # Get the input parameters
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb, Pe) = input_params

    # Prompt that there is no shear
    if shear_rate == 0.0:
        RuntimeWarning(f"Shear rate is zero!")

    # Calculate velocity profile at each frame and average over all frames
    binned_velocities_over_frames = np.array([lin_bin_stat(positions[i], velocities[i], box_length, num_bins=n_bins)[1] for i in range(len(positions))])
    binned_velocities = binned_velocities_over_frames.mean(axis=0)
    binned_velocities = binned_velocities[::-1] # flip them for some reason

    # Create the y_bins
    y_range = np.linspace(0.0 - 0.5 * box_length,  0.5 * box_length, n_bins,)
    binned_y = (y_range[:-1] + y_range[1:]) / 2.0 # take the center of the bin

    # Write the output to a file
    file = open("Velocityprofile"+fileout+".dat","w+")
    file.write("y   v\-(x)  v_real\n")
    for i in range(len(binned_y)):
        file.write(str(binned_y[i])+"   "+str(binned_velocities[i])+"   "+str(binned_y[i]*shear_rate/period)+"\n")
    file.close

    return

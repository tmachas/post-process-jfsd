import numpy as np
import freud


def calculate_msd(trajectory: np.ndarray[float], input_params: tuple, fileout: str):
    """
    Function to calculate the msd from the unwrapped trajectory
    
    Parameters
    -----------
    trajectory: (ndarray)
        The input trajectory
    input_params: (tuple)
        The input parameters
    fileout: (str)
        The name of the parent directory (for naming the output files) 

    """
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb, Pe) = input_params

    # Define the box dimensions (assuming a cubic box for simplicity)
    half_box_length = box_length / 2.0

    # Initialize an array to store the unwrapped trajectory
    unwrapped_trajectory = np.zeros_like(trajectory)
    unwrapped_trajectory[0] = trajectory[0]  # Start with the first frame as is

    # Unwrap the trajectory by checking for boundary crossings
    for t in range(1, trajectory.shape[0]):
        delta = trajectory[t] - trajectory[t - 1]
        
        # Apply the minimum image convention for each particle
        delta[delta > half_box_length] -= box_length  # Adjust if the displacement is > half the box length (positive direction)
        delta[delta < -half_box_length] += box_length  # Adjust if the displacement is < -half the box length (negative direction)

        # Update the unwrapped position
        unwrapped_trajectory[t] = unwrapped_trajectory[t - 1] + delta

    #np.save("unwrappedtrajectory",unwrapped_trajectory)

    # Initialize the MSD calculator
    msd_calculator = freud.msd.MSD()

    # Compute the MSD using the unwrapped trajectory
    msd_calculator.compute(unwrapped_trajectory)

    # Retrieve the mean squared displacement results
    msd = msd_calculator.msd

    file=open("FreudMSD"+fileout+".dat","w+") #storing the unwrappped MSD
    file.write("t/t\-(B)    MSD\n")
    for i in range(n_steps-1):
        file.write(str(time[i+1]/tb)+"   "+str(msd[i+1])+"\n")
    file.close

    return
import numpy as np
from scipy.special import gamma

from post_process_jfsd.utils import simulation_parameters, load_and_check
from post_process_jfsd.msd import calculate_msd


def msd_to_lve(fileout: str):
    """
    Function to calculate the Linear Viscoelastic spectrum from the MSD. If the MSD file is not found, it is calculated, provided the trajectory exists.

    Parameters
    -----------
    fileout: (str)
        The name of the parent directory
    """
    # Constants
    pi = np.pi
    a = 1

    # Load data (text format)
    try:
        data = np.loadtxt("MSD"+fileout+".dat", skiprows=1)
    except FileNotFoundError:
        print("MSD file not found. Calculating now...")

        trajectory, _, _, _ = load_and_check(False)
        input_params = simulation_parameters(trajectory)
        calculate_msd(trajectory, input_params, fileout)

        print("MSD calculated!")
        try:
            data = np.loadtxt("MSD"+fileout+".dat", skiprows=1)
        except FileNotFoundError:
            print("MSD file still not found. Something else is wrong. Abort!")
            exit()
    # Transpose the data to read them properly
    data = np.transpose(data)

    time = data[0]
    del_r2 = data[1]

    # Preallocate arrays
    alpha = []
    omega = []
    x_vals = []
    Gstar = []

    # Compute alpha, omega, x, and Gstar
    for i in range(1, len(time) - 1):
        log_ratio_r2 = np.log(del_r2[i + 1] / del_r2[i - 1])
        log_ratio_time = np.log(time[i + 1] / time[i - 1])
        alpha_val = log_ratio_r2 / log_ratio_time
        alpha.append(alpha_val)
        
        omega_val = 1 / time[i]
        omega.append(omega_val)
        
        x = 1 + alpha_val
        x_vals.append(x)
        
        Gstar_val = 1.0 / (pi * a * del_r2[i] * gamma(x))
        Gstar.append(Gstar_val)

    # Compute G' and G''
    Gp = [abs(G) * np.cos(pi * a_val / 2) for G, a_val in zip(Gstar, alpha)]
    Gdp = [abs(G) * np.sin(pi * a_val / 2) for G, a_val in zip(Gstar, alpha)]

    # Convert to NumPy arrays for easier handling
    omega = np.array(omega)
    Gp = np.array(Gp)
    Gdp = np.array(Gdp)

    # Write the output in a file
    file = open("LVEfromMSD"+fileout+".dat","w+")
    file.write("\g(w)   Gp   Gpp\n")
    for i in range(len(omega)):
        file.write(str(omega[i])+"   "+str(Gp[i])+"   "+str(Gdp[i])+"\n")
    file.close

    return

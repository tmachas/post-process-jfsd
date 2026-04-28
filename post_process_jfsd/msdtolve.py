import numpy as np
from numpy import ndarray as Array
from scipy.special import gamma

from post_process_jfsd.utils import simulation_parameters, load_and_check
from post_process_jfsd.msd import calculate_msd


def msd_to_lve(fileout: str) -> tuple[Array, Array, Array] :
    """
    Function to calculate the Linear Viscoelastic spectrum from the MSD. If the MSD file is not found, it is calculated, provided the trajectory exists.

    Parameters
    -----------
    fileout: (str)
        The name of the parent directory

    Returns
    -----------
    omega: (Array)
        The values of the angular frequency (normalized by tb)
    Gp: (Array)
        The normalized storage modulus values
    Gdp: (Array)
        The normalized loss modulus values
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
        columns = calculate_msd(trajectory, input_params, fileout)
        data = np.vstack(columns)
        print("MSD calculated!")
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

    return omega, Gp, Gdp

import numpy as np
from numpy import ndarray as Array
import freud
from scipy.integrate import simps


def gofr(trajectory: Array, frame: int, last_frame_index: int, input_params: tuple, N_gofr_bins: int, r_max: float) -> tuple[Array, Array]:
    """
    A function to calculate the radial distribution function for a given trajectory

    Parameters
    ------------

    trajectory: (Array)
        The input array
    frame: (int)
        Frame for which g(r) will be calculated
    last_frame_index: (int)
        The last non zero frame of the simulation
    input_params: (tuple)
        The simulation parameters
    N_gofr_bins: (int)
        Number of g(r) bins
    r_max: (float)
        Maximum r for g(r) calculation

    Returns
    ------------

    r_values: (Array)
        The radial distance values
    gofr: (Array)
        The calculated radial pdf
    """

    # Testing if input frame is out of range
    if frame > last_frame_index:
        raise ValueError(f"Selected frame is out of range. Last frame has index {last_frame_index}. Exiting...")

    positions = trajectory[frame]
    
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb) = input_params

    # Initialize the calculator, set the r_values and make the freud box
    gofr_calculator = freud.density.RDF(bins = N_gofr_bins, r_max = r_max)
    r_values = np.linspace(0, r_max, N_gofr_bins)
    box = freud.box.Box.cube(box_length)

    gofr_calculator.compute(system = (box,positions))

    gofr = gofr_calculator.rdf


    return r_values, gofr


def Sofk_from_gofr(r_values: Array, g_of_r: Array, input_params: tuple, py_theory_flag: bool): 
    """
    Compute static structure factor S(k) from g(r).
    
    Parameters
    ----------
    r_values : Array
        Radial distances 
    g_r : Array
        Radial distribution function values at r
    input_params: tuple
        A tuple containing the simulation parameters
    frame: int
        The frame for which S(k) will be calculated
        
    Returns
    -------
    k_values: Array
        The inverse space grid points
    S_k : Array
        Structure factor values corresponding to k_values
    """
    
    def py_structure_factor(q: float, phi: float):
        """Fuction that returns the Percus-Yevik structure factor for a given volume fraction (taken from https://en.wikipedia.org/wiki/Percus%E2%80%93Yevick_approximation)
        
        Parameters
        ----------
        q: Array
            The inverse space grid points
        phi: float
            The volume fraction of the system
        
        Returns
        --------
        S(k): Array
            The ideal structure factor of hard spheres for volume fraction phi 
        """

        a = (1 + 2 * phi) ** 2 / (1 - phi) ** 4
        b = -6 * phi * (1 + phi / 2) ** 2 / (1 - phi) ** 4
        c = phi / 2 * (1 + 2 * phi) ** 2 / (1 - phi) ** 4

        A = 2 * q
        A2 = A * A

        G = (
            a / A2 * (np.sin(A) - A * np.cos(A))
            + b / (A * A2) * (2 * A * np.sin(A) + (2 - A2) * np.cos(A) - 2)
            + c / (A**5) * (
                -A**4 * np.cos(A)
                + 4 * ((3 * A2 - 6) * np.cos(A) + A * (A2 - 6) * np.sin(A) + 6)
            )
        )

        return 1 / (1 + 24 * phi * G / A)
    
    S_k = []

    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb) = input_params

    # Testing if zero belongs to the r_values (if it belongs, it will be the first one) and deleting it
    if r_values[0] == 0.0:
        r_values = np.delete(r_values, 0)
        g_of_r = np.delete(g_of_r, 0)
    
    # Fix the k values
    k_values = 2.0*np.pi/r_values

    # Do the spherical fourier transform numerically to get the S(k)
    for k in k_values:
        integrand = (g_of_r - 1.0) * np.sin(k*r_values) / (k*r_values) * r_values**2
        integral = 4.0 * np.pi * N / (box_length**3) * simps(integrand, r_values)
        S_k.append(1.0 + integral)

    structure_factor = np.array(S_k)

    # Write the output in a file
    if py_theory_flag:
        py_sofk = py_structure_factor(k_values, 4.0/3.0*N*np.pi/(box_length**3))

    else:
        py_sofk = None

    return r_values, structure_factor, py_sofk
    
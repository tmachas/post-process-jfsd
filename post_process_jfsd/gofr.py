import numpy as np
from numpy import ndarray as Array
import freud

def gofr(trajectory: Array, frame: int, last_frame_index: int, input_params: tuple, N_gofr_bins: int, r_max: float, fileout: str) -> tuple[Array, Array]:
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
    fileout: (str)
        The name of the parent directory

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

    # Write the output in a file
    file = open("gofr"+fileout+".dat","w+")
    file.write("r/R   g(r)\n")
    for i in range(len(r_values)):
        file.write(str(r_values[i])+"   "+str(gofr[i])+"\n")
    file.close

    return r_values, gofr
    
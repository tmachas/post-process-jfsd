import numpy as np
from numpy import ndarray as Array
import freud

from post_process_jfsd.utils import calculate_distances


def average_bonds_number(trajectory: Array, input_params: tuple, attr_range = 0.1) -> tuple[Array, Array, Array]:
    """
    A function to calculate the averaged over particles number of bonds, for all frames

    Parameters
    -------------
    trajectory: (Array)
        The particles trajectory array of shape (n_steps, N, 3)
    input_params: (tuple)
        A tuple containing the input parameters
    attr_range: (float)
        The attraction range of the depletion potential

    Returns
    --------------
    time/tb: (Array)
        The time intervals normalized by the brownian time
    av_bonds: (Array)
        The average number of bonds for every time step
    std_bonds: (Array)
        The standard deviation of the number of bonds for every time step
    """
    bonded_radius = 2.0*(1.0 + attr_range)

    # Untuple parameters
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb) = input_params

    # Initialize the return arrays
    av_bonds = np.zeros((n_steps))
    std_bonds = np.zeros((n_steps))
    
    for step in range(n_steps):
        positions = trajectory[step]
        distance_vectors = calculate_distances(positions, N, box_length)
        norms = np.linalg.norm(distance_vectors, axis = 2)

        # Find which particles are bonded and average
        bonded = np.where(norms < bonded_radius, 1, 0)
        bonds_per_particle = np.sum(bonded, axis=1)

        # Remove the self bond
        bonds_per_particle = bonds_per_particle - 1

        # and average
        av_bonds[step] = np.average(bonds_per_particle)
        std_bonds[step] = np.std(bonds_per_particle)
        
    return time/tb, av_bonds, std_bonds


def average_voronoi_volume(trajectory: Array, input_params: tuple) -> tuple[Array, Array]:
    """
    A function to calculate the averaged over particles voronoi volume for all time frames

    Parameters
    -------------
    trajectory: (Array)
        The particles trajectory array of shape (n_steps, N, 3)
    input_params: (tuple)
        A tuple containing the input parameters

    Returns
    --------------
    time/tb: (Array)
        The time intervals normalized by the brownian time
    av_voronoi_volume: (Array)
        The average voronoi volume for each time frame
    """

    # Untuple parameters
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb) = input_params

    # Initialize the voronoi calculator
    voronoi_volumes = np.zeros((n_steps, N))
    box = freud.box.Box.cube(box_length)
    voronoi = freud.locality.Voronoi()
        
    for step in range(n_steps):
        positions =  trajectory[step]

        voronoi_calculator = voronoi.compute(system = (box, positions))
        voronoi_volumes[step] = voronoi_calculator.volumes

    av_voro_volume = np.average(voronoi_volumes, axis=1)

    return time/tb, av_voro_volume
import numpy as np
from numpy import ndarray as Array
from jax import jit
import jax.numpy as jnp

from post_process_jfsd.utils import log_bin_stat, calculate_distances

def calculate_particle_stress_correction(trajectory: Array, input_params: tuple, raw_stress_flag: bool, spring_const = 2500.0) -> tuple[Array, Array]:
    """
    Function to calculate the <xF> term of the stress tensor and output it seperately

    Parameters
    ------------
    trajectory: (Array)
        The positions of the particles for every frame
    input_params: (tuple)
        The simulation input parameters
    raw_stress_flag: (bool)
        Flag whether the only-over-particle-averaged stress is outputed
    spring_const: (float)
        The spring constant of the harmonic hard sphere potential

    Returns
    -------------
    binned_times*shear_rate: (Array)
        The binned strain values
    binned_stress_xy: (Array)
        The dimensionless xy component of the particle stress tensor
    """
    @jit
    def calculate_xF_for_frame(positions: Array) -> Array:
        """
        Seperate routine to calculate the xF tensor for a given frame. Done so it can be jit-ed

        Parameters
        -------------
        positions: (Array)
            Positions of particles for a given frame
        
        Returns
        -------------
        S: (Array)
            The <xF> tensor of shape (N, 3, 3)
        """
        distance_vectors = calculate_distances(positions, N, box_length)

        # Compute Euclidean norms for each distance vector
        norms = jnp.linalg.norm(distance_vectors, axis=2)  # shape: (N, N)

        # Broadcast norms into shape (N, N, 3)
        norm_matrix = jnp.repeat(norms[:, :, jnp.newaxis], 3, axis=2)

        norm_matrix = jnp.where(norm_matrix == 0.0, jnp.inf, norm_matrix)
        
        # Calculate forces
        Fp = jnp.zeros((N, N, 3))
        Fp = k * (1-sigma/norm_matrix) * distance_vectors / norm_matrix
        
        Fp = jnp.where(norm_matrix < sigma, Fp, 0.0)
        
        # Calculate the xF term
        stress_tensor_temp = jnp.zeros((N, N, 3, 3))
        
        stress_tensor_temp = distance_vectors[..., :, jnp.newaxis] * Fp[..., jnp.newaxis, :]
        S_p = jnp.sum(stress_tensor_temp, axis=1)

        # Average and normalize
        S = jnp.average(S_p, axis=0) * N / (box_length)**3 / kT
        
        return S
    
    
    # Untuple parameters
    (n_steps, N, dt, period, time, kT, shear_rate, box_length) = input_params

    # Potential characteristics
    k = spring_const / dt
    sigma = 2. * (1.001)

    # Initialize stress tensor
    stress_tensor = np.zeros((n_steps, 3, 3))

    for step in range(n_steps):
        
        positions = trajectory[step]
        stress_tensor[step] = calculate_xF_for_frame(positions)

    # Reshape just for my convenience
    stress_tensor_reshaped = np.reshape(stress_tensor, (n_steps, 9))

    # If raw_stress_flag == True, return the only particle averaged stress. Else return the bin averaged
    if raw_stress_flag:
        return time*shear_rate, np.transpose(stress_tensor_reshaped)[1]
    
    else:
        binned_times, binned_stress_xy = log_bin_stat(time, np.transpose(stress_tensor_reshaped)[1], num_bins=80)

        return binned_times*shear_rate, binned_stress_xy



def caclulate_average_stress(stresslet: Array, input_params: tuple, raw_stress_flag: bool, N_stress_bins: int) -> tuple[Array, Array]:
    """
    A function to calculate the logarithmic binned average of the stresslet. There is also option to save the only-particle-averaged stresslet

    Parameters
    ----------

    stresslet: (ndarray)
        The input stresslet. Should be shape (N_steps, N, 5)
    input_params: (tuple)
        The simulation parameters
    raw_stress_flag: bool
        Flag the calculation of the only-particle-averaged stresslet
    N_stress_bins: (int)
        The number of bins for the stress average

    Returns
    -------------
    binned_times/tb*Pe: (Array)
        The strain values 
    binned_stress_xy: (Array)
        The dimensionless average xy component of the stresslet

    Notes
    ----------
    The stress tensor elements are correlated with the stresslet through the relations:
    s_xx = S0
    s_xy = S1
    s_xz = S2
    s_yy = S3
    s_yz = S4
    + the zero trace of the stress tensor
    """

    # Get the simulation parameters
    (n_steps, N, dt, period, time, kT, shear_rate, box_length) = input_params
    
    #Take ensemble average
    av_stresslet = np.average(stresslet, 1)


    if raw_stress_flag == True: # store the only-particle averaged stress
        raw_stresslet = av_stresslet * N / (box_length**3) / kT # Translate the stresslet to the stress tensor and normalize

        return time*kT, time*shear_rate, raw_stresslet[:,1], raw_stresslet[:,0], raw_stresslet[:,2], 0.0 - raw_stresslet[:,0] - raw_stresslet[:,2]

    #Prepare the stresslets for the binning
    xy_stresslet = av_stresslet[:,[1]].ravel()
    xx_stresslet = av_stresslet[:,[0]].ravel()
    yy_stresslet = av_stresslet[:,[3]].ravel()
    zz_stresslet = 0.0 - xx_stresslet - yy_stresslet

    #Calculate the binned stresslet for every component
    binned_times, binned_stresslet_xy = log_bin_stat(time, xy_stresslet, num_bins=N_stress_bins)
    _, binned_stresslet_xx = log_bin_stat(time, xx_stresslet, num_bins=N_stress_bins)
    _, binned_stresslet_yy = log_bin_stat(time, yy_stresslet, num_bins=N_stress_bins)
    _, binned_stresslet_zz = log_bin_stat(time, zz_stresslet, num_bins=N_stress_bins)

    #Trasnlate the stresslet to stress tensor using the particle number density and normalize
    binned_stresslet_xy = binned_stresslet_xy * N / (box_length**3) / kT
    binned_stresslet_xx = binned_stresslet_xx * N / (box_length**3) / kT
    binned_stresslet_yy = binned_stresslet_yy * N / (box_length**3) / kT
    binned_stresslet_zz = binned_stresslet_zz * N / (box_length**3) / kT

    return binned_times*kT, binned_times*shear_rate, binned_stresslet_xy, binned_stresslet_xx, binned_stresslet_yy, binned_stresslet_zz
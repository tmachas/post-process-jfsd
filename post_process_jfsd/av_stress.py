import numpy as np
from numpy import ndarray as Array

from post_process_jfsd.utils import log_bin_stat


def calculate_particle_stress_correction(trajectory: Array, input_params: tuple, raw_stress_flag: bool, fileout: str) -> tuple[Array, Array]:
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
    fileout: (str)
        The name of the parent directory

    Returns
    -------------
    binned_times*shear_rate: (Array)
        The binned strain values
    binned_stress_xy: (Array)
        The dimensionless xy component of the particle stress tensor
    """
    # Untuple parameters
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb, Pe) = input_params

    # Potential characteristics
    k = 2500 / dt
    sigma = 2. * (1.001)

    # Initialize stress tensor
    stress_tensor = np.zeros((n_steps, 3, 3))

    for step in range(n_steps):

        # Calculate the particle distance vectors (only the lower triangular part)
        distance_vectors = np.zeros((N, N, 3))
        norm_matrix = np.zeros((N, N, 3))
        
        positions = trajectory[step]
        distance_vectors = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape: (N, N, 3)

        # Compute Euclidean norms for each distance vector
        norms = np.linalg.norm(distance_vectors, axis=2)  # shape: (N, N)

        # Broadcast norms into shape (N, N, 3)
        norm_matrix = np.repeat(norms[:, :, np.newaxis], 3, axis=2)

        norm_matrix = np.where(norm_matrix == 0.0, np.inf, norm_matrix)
        
        # Calculate forces
        Fp = np.zeros((N, N, 3))
        Fp = k * (1-sigma/norm_matrix) * distance_vectors / norm_matrix
        
        Fp = np.where(norm_matrix < sigma, Fp, 0.0)
        
        # Calculate the xF term
        stress_tensor_temp = np.zeros((N, N, 3, 3))
        
        stress_tensor_temp = distance_vectors[..., :, np.newaxis] * Fp[..., np.newaxis, :]
        S_p = np.sum(stress_tensor_temp, axis=1)

        # Average and normalize
        S = np.average(S_p, axis=0) * N / (box_length)**3 / kT
        
        stress_tensor[step] = S


    # Reshape just for my convenience
    stress_tensor = np.reshape(stress_tensor, (n_steps, 9))

    binned_times, binned_stress_xy = log_bin_stat(time, np.transpose(stress_tensor)[1], num_bins=80)

    file = open("ParticleStressaveraged"+fileout+".dat","w+")
    file.write("\g(g)   \g(s)\-(xy)\n")
    for i in range(len(binned_times)):
        file.write(str(binned_times[i] * shear_rate)+"   "+str(binned_stress_xy[i])+"\n")
    file.close

    if raw_stress_flag:
        file = open("ParticleStress"+fileout+".dat","w+")
        file.write("\g(g)   \g(s)\-(xy)\n")
        for i in range(len(time)):
            file.write(str(time[i]*shear_rate)+"   "+str(np.transpose(stress_tensor)[1][i])+"\n")
        file.close

    return time*shear_rate, binned_stress_xy



def caclulate_average_stress(stresslet: Array, input_params: tuple, raw_stress_flag: bool, N_stress_bins: int, fileout: str) -> tuple[Array, Array]:
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
    fileout: (str)
        The name of the parent directory, for naming the output file

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
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb, Pe) = input_params
    
    #Take ensemble average
    av_stresslet = np.average(stresslet, 1)


    if raw_stress_flag == True: # store the only-particle averaged stress

        raw_stresslet = av_stresslet * N / (box_length**3) / kT # Translate the stresslet to the stress tensor and normalize

        file4 = open("AVST"+fileout+"raw.dat","w+")
        file4.write("t/t\-(B)   \g(g)   \g(s)\-(xy)   \g(s)\-(xx)   \g(s)\-(yy)   \g(s)\-(zz)\n")
        for i in range(n_steps):
            file4.write(str(time[i]/tb)+"   "+str(time[i]/tb*Pe)+"   "+str(raw_stresslet[i][1])+"   "+str(raw_stresslet[i][0])+"   "+str(raw_stresslet[i][2])+"   "+str(0.0 - raw_stresslet[i][0] - raw_stresslet[i][2])+"\n")
        file4.close

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

    
    # Save the averaged stresslet
    file3 = open("AVST"+fileout+".dat","w+") #storing the stress tensor
    file3.write("t/t\-(B)   \g(g)   \g(s)\-(xy)   \g(s)\-(xx)   \g(s)\-(yy)   \g(s)\-(zz)\n")
    for i in range(len(binned_times)):
        file3.write(str(binned_times[i]/tb)+"   "+str(binned_times[i]/tb*Pe)+"   "+str(binned_stresslet_xy[i])+"   "+str(binned_stresslet_xx[i])+"   "+str(binned_stresslet_yy[i])+"   "+str(binned_stresslet_zz[i])+"\n")
    file3.close

    return binned_times/tb*Pe, binned_stresslet_xy
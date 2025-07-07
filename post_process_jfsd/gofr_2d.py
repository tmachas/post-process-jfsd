import numpy as np
from numpy import ndarray as Array
from matplotlib import pyplot as plt
import cmcrameri.cm as cmc


def gofxy_for_frame(trajectory: Array, 
                    x_bins: Array, 
                    y_bins: Array, 
                    Xmax: float, Ymax: float, 
                    N: int, 
                    slice_width: float, 
                    frame: int) -> Array:
    """
    Function to calculate the gofxy for a specific frame

    Parameters
    -----------
    
    frame: (int)
           Frame indice to be calculated

    Returns
    -----------
    gofxy: (ndarray)
    """
    positions = trajectory[frame]

    # Calculate interparticle distances
    distance_vectors = np.zeros((N, N, 3))
    distance_vectors = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape: (N, N, 3)

    # Select only the particles within the slice and remove the self contribution
    indices = np.where(np.abs(distance_vectors[:,:,2]) < slice_width, 1, 0)
    indices += -1 * np.eye(N, dtype=int)

    # Setting all the x and y distances outside the slice to a number bigger than the binning box
    x_distances = np.where(indices, distance_vectors[:,:,0], Xmax + 10)
    y_distances = np.where(indices, distance_vectors[:,:,1], Ymax + 10)

    # g of xy for every particle and average the images
    gofxy_set = np.array([np.histogram2d(x_distances[i], y_distances[i], bins = (x_bins, y_bins), density=True)[0] for i in range(N)])
    gofxy = gofxy_set.mean(axis=0)
    
    return gofxy


def gofxy_image(
    trajectory: Array, 
    input_params: tuple, 
    last_frame_index: int, 
    frame: int, 
    subtract_rest: bool,
    fileout: str,
    slice_width: float,
    N_gofxy_bins: int,
    Xmax: float,
    Ymax: float)  -> Array :

    """
    Create the image of the xy projection of the g(r) for a specific time frame. There is the option to subtract from it the g(r) at rest (of the first frame)

    Parameters
    -----------
    trajectory: (ndarray)
        The input trajectory
    input_params: (tuple)
        Tuple containing all simulation parameters
    last_frame_index: (int)
        The index of the last non-zero frame of the trajectory (in case of premature ending)
    frame: (int)
        Frame for which g(r) is calculated
    subtract_rest: (bool)
        Choose whether to subtract the zeroth frame
    fileout: (str)
        The name of the parent directory
    slice_width: (float)
        The width of the z-axis slice for which the xy average is calculated
    N_gofxy_bins: (int)
        The number of bins for each axis on the g(r) image
    Xmax: (float)
        The maximum x value of the image (total image ranges [-Xmax, Xmax])
    Ymax: (float)
        The maximum y value of the image (total image ranges [-Ymax, Ymax])

    Returns
    -----------
    gofxy_to_be_plotted: (Array)
        The gofxy values for the given frame; has dimensions (n_bins, n_bins)
    """
    # Testing if input frame is out of range
    if frame > last_frame_index:
        raise ValueError(f"Selected frame is out of range. Last frame has index {last_frame_index}. Exiting...")

    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb, Pe) = input_params

    # Testing if Xmax and Ymax are bigger than box size
    if (Xmax > box_length / 2.) or (Ymax > box_length / 2.) :
        raise ValueError(f"Xmax and Ymax cannot be larger than half of the box size. Try values larger than {box_length*0.5}")

    # Create bins
    x_bins = np.linspace(-Xmax, Xmax, N_gofxy_bins)
    y_bins = np.linspace(-Ymax, Ymax, N_gofxy_bins)


    # Define the new x and y edges
    xedges, yedges = np.delete(x_bins,0), np.delete(y_bins,0)

    # Plot results
    X, Y = np.meshgrid(xedges, yedges)

    if subtract_rest==True:
        gofxy_to_be_plotted = gofxy_for_frame(trajectory, x_bins, y_bins, Xmax, Ymax, N, slice_width, frame) - gofxy_for_frame(trajectory, x_bins, y_bins, Xmax, Ymax, N, slice_width, frame = 0)
        plt.pcolormesh(X, Y, gofxy_to_be_plotted, cmap=cmc.berlin)
        title_add = "_zeroth_frame_subtracted"
    else:
        gofxy_to_be_plotted = gofxy_for_frame(trajectory, x_bins, y_bins, Xmax, Ymax, N, slice_width, frame)
        plt.pcolormesh(X, Y, gofxy_to_be_plotted)
        title_add = ""
    plt.colorbar()
    plt.title(fileout+f" Frame = {frame} " + title_add)

    # Save the image
    plt.savefig("gofxy"+fileout+"frame"+str(frame)+title_add+".png")

    return gofxy_to_be_plotted
import toml

from post_process_jfsd.utils import dir_name, simulation_parameters, load_and_check
from post_process_jfsd.msd import calculate_msd
from post_process_jfsd.av_stress import caclulate_average_stress, calculate_particle_stress_correction
from post_process_jfsd.npy_to_xyz import npy_to_xyz
from post_process_jfsd.gofr_2d import gofxy_image
from post_process_jfsd.gofr import gofr
from post_process_jfsd.velocity_profile import vel_profile
from post_process_jfsd.msdtolve import msd_to_lve




def main():
    
    print("JFSD post processing script\n")
    
    print("Reading the settings file...")
    try:
        with open('post_process_settings.toml', 'r') as f:
            settings_file = toml.load(f)
            print(f"Settings obtained from {f.name}\n")
        
        basic_process = bool(settings_file['basic']['just_basic_calculation'])

    except FileNotFoundError:
        print("There is no post processing file! Assuming basic post processing\n")
        basic_process = True

    if basic_process == True:
        msd_flag = True
        msd_windowed = True
        
        av_stress_flag = True
        N_stress_bins = 80
        raw_stress_flag = False
        xF_flag = False
        
        gofxy_flag = False

        gofr_flag = False

        v_profile_flag = False
        
        ovito_flag = False

        lve_flag = False
    else:
        msd_flag = bool(settings_file['MSD']['MSD_calculation'])
        msd_windowed_flag = bool(settings_file['MSD']['windowed_msd'])
        
        av_stress_flag = bool(settings_file['Stresses']['Binned_stress_average_calculation'])
        N_stress_bins = int(settings_file['Stresses']['N_stress_bins'])
        raw_stress_flag = bool(settings_file['Stresses']['Raw_stress_output'])
        xF_flag = bool(settings_file['Stresses']['particle_stress_correction'])

        gofxy_flag = bool(settings_file['gofxy']['gofxy_calculation'])
        gofxy_frame = int(settings_file['gofxy']['frame'])
        gofxy_slice_width = float(settings_file['gofxy']['slice_width'])
        gofxy_subtract_rest_flag = bool(settings_file['gofxy']['subtract_rest'])
        N_gofxy_bins = int(settings_file['gofxy']['N_gofxy_bins'])
        Xmax = float(settings_file['gofxy']['Xmax'])
        Ymax = float(settings_file['gofxy']['Ymax'])

        gofr_flag = bool(settings_file['gofr']['gofr_calculation'])
        gofr_frame = int(settings_file['gofr']['frame'])
        N_gofr_bins = int(settings_file['gofr']['N_gofr_bins'])
        gofr_r_max = float(settings_file['gofr']['r_max'])

        v_profile_flag = bool(settings_file['velocity_profile']['v_profile_calculation'])
        v_profile_bins = int(settings_file['velocity_profile']['N_bins'])

        ovito_flag = bool(settings_file['ovito_file']['xyz_file'])

        lve_flag = bool(settings_file['MSD_to_LVE']['lve_calculation'])

    # Load the input files
    (trajectory, stresslet, velocities, last_frame_index) = load_and_check(av_stress_flag, v_profile_flag)

    # Load the simulation parameters
    input_params = simulation_parameters(trajectory)
    (n_steps, N, dt, period, time, kT, shear_rate, box_length, tb) = input_params

    print("Post processing parameters")
    print("-------------------------")
    print(f"MSD calculation: {msd_flag}")
    if msd_flag:
        print(f"Windowed msd: {msd_windowed_flag}")
    print("")
    if av_stress_flag:
        print(f"Stress calculation: {av_stress_flag} with Pe = {shear_rate*tb}")
        print(f"    Raw stress: {raw_stress_flag}")
        print(f"    <xF> correction: {xF_flag}")
    else:
        print(f"Stress calculation: {av_stress_flag}")
    print("")
    print(f"g(r) calculation is: {gofr_flag}")
    if gofr_flag:
        print(f"Frame = {gofr_frame}")
    print("")
    print(f"g(r) on xy plane calculation: {gofxy_flag}")
    if gofxy_flag:
        print(f"Frame = {gofxy_frame}")
        print(f"Subtract zeroth frame: {gofxy_subtract_rest_flag}")
    print("")
    print(f"Velocity profile calculation: {v_profile_flag}")
    print("")
    print(f"Ovito file output: {ovito_flag}")
    print("")
    print(f"LVE spectrum calculation: {lve_flag}")
    print("-------------------------")


    # Get the directory name
    fileout = dir_name()

    # Calculate the msd
    if msd_flag:
        print("Calculating MSD...")
        calculate_msd(trajectory, input_params, msd_windowed_flag, fileout)

    if av_stress_flag:
        print("Calculating stresses...")
        caclulate_average_stress(stresslet, input_params, raw_stress_flag, N_stress_bins, fileout)
        if xF_flag:
            calculate_particle_stress_correction(trajectory, input_params, raw_stress_flag, fileout)

    if gofr_flag:
        print("Calculating g(r)...")
        gofr(trajectory, gofr_frame, last_frame_index, input_params, N_gofr_bins, gofr_r_max, fileout)
    
    if gofxy_flag:
        print("Calculating g(r) on xy plane...")
        gofxy_image(trajectory, input_params, last_frame_index, gofxy_frame, gofxy_subtract_rest_flag, fileout, gofxy_slice_width, N_gofxy_bins, Xmax, Ymax)

    if v_profile_flag:
        print("Calculating velocity profile...")
        vel_profile(trajectory, velocities, input_params, v_profile_bins, fileout)

    if ovito_flag:
        print("Writing ovito file...")
        npy_to_xyz(trajectory, fileout)

    if lve_flag:
        print("Calculating LVE spectrum...")
        msd_to_lve(fileout)
    
    print("Done!")

    return



if __name__ == "__main__":
    main()
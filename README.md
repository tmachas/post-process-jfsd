# post_process_jfsd

This package contains various post processing routines, tailored for the Jax Fast Stokesian Dynamics implementation of the stokesian dynamics colloidal simulation method (https://github.com/torrewk/Python-Jax-Fast-Stokesian-Dynamics)

Currently includes:
- Mean square displacement
- Stress tensor calculation from the hydrodynamic stresslet (with correction of the interparticle <xF> term)
- Calculation of the Linear Viscoelastic spectrum from the Mean square displacement using the Generalized Einstein equation
- Calculation of the radial distribution function g(r)
- Calculation of the xy projection of the g(r)
- Creation of an ovito/vmd compatible .xyz file for the particle trajectories

## Installation

```bash
cd post_process_jfsd
pip install .
```

## Usage

After installing, run:

```bash
post_process_jfsd
```
in the directory containing the input.toml file and the simulation outputs (trajectory.npy, stresslet.npy).

To specify the parameters of the post processing, as well as which post processing routines will be executed, paste the post_process_settings.toml file in the simulation output directory and modify it accordingly.
Else, only the MSD and average stress is calculated by default.

## Requirements

- Python >= 3.10
- numpy
- matplotlib
- scipy
- freud_analysis
- toml
- cmcrameri


## Contact
Athanasios Machas, 
University of Crete and Foundation for Research and Technology Hellas, Greece
amachas@materials.uoc.gr

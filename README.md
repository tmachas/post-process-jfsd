# post_process_jfsd

This package contains various post processing routines, tailored for the Jax Fast Stokesian Dynamics implementation of the stokesian dynamics colloidal simulation method

Currently includes:
- Mean square displacement
- Stress tensor calculation from the hydrodynamic stresslet (with correction of the interparticle <xF> term)
- Calculation of the Linear Viscoelastic spectrum from the Mean square displacement using the Generalized Einstein equation
- Calculation of the xy projection of the g(r)
- Creation of an ovito/vmd compatible .xyz file for the particle trajectories

## Installation

```bash
cd post_process_jfsd
pip install .
```

Or, for development mode:

```bash
pip install -e .
```

## Usage

After installing, run:

```bash
post_process_jfsd
```

This executes the main processing workflow.

## Requirements

- Python >= 3.10
- numpy
- matplotlib
- scipy
- freud_analysis
- toml
- cmcrameri

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

## Contact
Athanasios Machas
University of Crete and Foundation for Research and Technology Hellas, Greece
amachas@materials.uoc.gr

# DM² for SiC: A Diffusion Model for Silicon Carbide Glass Generation

This project adapts the original [DM²](https://github.com/digital-synthesis-lab/DM2) framework to generate amorphous Silicon Carbide (SiC) glass structures. It is a complete workflow, from generating training data using LAMMPS simulations to training a conditional diffusion model and generating new SiC structures.

## Folder Structure

The project is organized into three main directories:

-   `SiC LAMMPS Simulations`: Contains scripts and input files to run LAMMPS simulations and generate the training data.
-   `DM2_SiC Model Training`: Contains the Python script and necessary data to train the conditional diffusion model.
-   `DM2_SiC generation`: Contains the Python script to generate new SiC glass structures using the trained model.

## `SiC LAMMPS Simulations`

This directory contains the necessary files to generate amorphous SiC structures using LAMMPS. The workflow is automated using the `run_workflow.py` script, which calls other scripts and LAMMPS input files.

### Contents:

-   `run_workflow.py`: The main script to automate the LAMMPS simulation workflow. It runs a series of simulations to melt and quench SiC, generating amorphous structures.
-   `run_multi_batch.sh`: A shell script to submit multiple simulation jobs in parallel.
-   `pack_SiC.inp`: A packmol input file to create the initial random distribution of Si and C atoms.
-   `in.minimize_auto`: LAMMPS input script for energy minimization of the initial structure.
-   `in.SiC_melt_quench_auto`: LAMMPS input script for the melt-quench simulation.
-   `SiC.vashishta`: The Vashishta potential file for SiC, used to model the interatomic interactions.
-   `si.xyz`, `c.xyz`: Coordinate files for single Si and C atoms, used by `pack_SiC.inp`.
-   `Example_SiC_Simulation_output`: A directory containing an example of the output from a LAMMPS simulation.

### Usage:

1.  Modify `run_workflow.py` to set the desired simulation parameters (e.g., temperature, cooling rate).
2.  Run the script: `python run_workflow.py`
3.  The output data files will be saved in new directories with the format `run_<run_number>_<temperature>_<cooling_rate>`.

## `DM2_SiC Model Training`

This directory is for training the conditional diffusion model on the generated SiC data. The training process is conditioned on the cooling rate, allowing for the generation of structures with properties corresponding to a specific cooling rate.

### Contents:

-   `denoise_train_conditional.py`: The main Python script to train the diffusion model. It uses a UNet architecture and is conditioned on the cooling rate.
-   `simu_data`: A directory containing the SiC data files generated from the LAMMPS simulations.
-   `Model_v0 results`: A directory where the trained model checkpoints and training logs are saved.

### Usage:

1.  Place your LAMMPS-generated SiC `.data` files into the `simu_data` directory.
2.  Run the training script: `python denoise_train_conditional.py`
3.  The trained model will be saved in the `Model_v0 results` directory.

## `DM2_SiC generation`

This directory is used to generate new SiC glass structures using the trained conditional diffusion model.

### Contents:

-   `denoise_generate_conditional.py`: The Python script to generate new SiC structures. It loads a trained model and generates structures conditioned on a specified cooling rate.
-   `inital_data`: A directory containing an initial data file to start the generation process from.
-   `output`: The directory where the generated structures are saved in `.extxyz` format.
-   `LAMMPS_rdf.png`, `Denoise_output_rdf.png`: Example plots of the radial distribution function (RDF) for comparison between LAMMPS and generated structures.

### Usage:

1.  Make sure you have a trained model from the `DM2_SiC Model Training` step.
2.  Modify `denoise_generate_conditional.py` to specify the path to the trained model, the initial data file, and the desired cooling rate for generation.
3.  Run the script: `python denoise_generate_conditional.py`
4.  The generated structure will be saved in the `output` directory.

## Workflow Summary

1.  **Data Generation:** Use the scripts in `SiC LAMMPS Simulations` to generate a dataset of amorphous SiC structures with varying cooling rates.
2.  **Model Training:** Use the generated data and the script in `DM2_SiC Model Training` to train a conditional diffusion model.
3.  **Structure Generation:** Use the trained model and the script in `DM2_SiC generation` to generate new amorphous SiC structures with desired cooling rates.

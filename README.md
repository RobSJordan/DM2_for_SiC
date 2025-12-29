# DM² for SiC: A Diffusion Model for Silicon Carbide Glass Generation

This project adapts the original [DM²](https://github.com/digital-synthesis-lab/DM2) framework to generate amorphous Silicon Carbide (SiC) glass structures. It is a complete workflow, from generating training data using LAMMPS simulations to training a conditional diffusion model and generating new SiC structures.

## Overview

This project follows the workflow desribed within the original DM2 paper (https://arxiv.org/abs/2507.05024) to train a model for the generation of amorphous SiC structures. Due to computational limiations, a number of modifications were made to the original code and training strategy. Only the conditonal training was carried out. MD simulations and model training were carried out on a workstation with an Intel Core Ultra 265K with 96GB RAM and a NVIDIA GeForce RTX 3080 GPU with 12GB VRAM.

1. 15 amorphous SiC structures were generated, using different cooling rates during the melt-quench process. Cooling rates of 100k, 10k, and 1k / ps were calulated. 0.1k was ommitted due to limited computational resources.
2. A diffusion model was trained using the DM2 framework, with a batch-size of 1. Although there was some unstability in the training, rdf analysis of the generated structures shows good overlap with LAMMPS generated structures. (see below)

# rdf plot of example LAMMPS generated structure
![alt text](<DM2_SiC generation/Denoise_output_rdf.png>)
# rdf plot of example diffusion generated structure
![alt text](<DM2_SiC generation/LAMMPS_rdf.png>)

3. Potential next steps include
-   Inclusion of a gradient accumulation step in the training process to improve stability with small batch sizes due to the limited VRAM available.


## Folder Structure

The project is organized into three main directories:

-   `SiC LAMMPS Simulations`: Contains scripts and input files to run LAMMPS simulations and generate the training data.
-   `DM2_SiC Model Training`: Contains the Python script and necessary data to train the conditional diffusion model.
-   `DM2_SiC generation`: Contains the Python script to generate new SiC glass structures using the trained model.

## Workflow Summary

1.  **Data Generation:** Use the scripts in `SiC LAMMPS Simulations` to generate a dataset of amorphous SiC structures with varying cooling rates.
2.  **Model Training:** Use the generated data and the script in `DM2_SiC Model Training` to train a conditional diffusion model.
3.  **Structure Generation:** Use the trained model and the script in `DM2_SiC generation` to generate new amorphous SiC structures with desired cooling rates.

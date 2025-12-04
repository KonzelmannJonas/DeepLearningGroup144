# Solving the 1D Burgers' equation with PINNs - Group 144 

This repo contains the code for the project of the group 144 for Deep Learning 02456 at DTU (Fall 2025).

## Folder structure
Most folders should be self-explainatory, with these additions:

- `notebook_burgers_eq_1d`: notebook explaining the default PINN
- `template_burgers_eq_1d`: template for all the methods, using the default PINN
- `paper_burgers_eq_2d`: implements a paper that solves the Burgers' equation in 2d. This was not discussed in our report. 

## Usage
- `pinn.py`: contains PINN class and all methods needed
- `train.py`: runs this script to train the PINN
- `plot.py`: used to plot the PINN

## General
- `train.py` sometimes already plots and saves the results. This varies between the methods. 
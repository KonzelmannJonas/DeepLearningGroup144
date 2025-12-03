"""
Training Script for KAN-PINN (Physics-Informed Neural Network).

This script orchestrates the training process for the KAN-based PINN solving 
the Burgers' equation. It is designed to be executed from the command line, 
making it ideal for submitting batch jobs in High-Performance Computing (HPC) 
environments.

Usage:
    python train_KAN.py --adam 1000 --lbfgs 4000 --name "Experiment_01"

Arguments:
    --adam  : Number of epochs for the Adam optimizer (Phase 1).
    --lbfgs : Maximum iterations for the LBFGS optimizer (Phase 2).
    --name  : Identifier for the experiment (used for saving logs and models).

Author: Group 144
Date: 2025
"""

import argparse
import sys
import os

# Import the PINN class from the main model file
# Ensure 'pinn_KAN.py' is in the same directory or in the PYTHONPATH
try:
    from pinn_KAN import PINN 
except ImportError:
    sys.exit("Error: Could not import 'PINN' from 'pinn_KAN.py'. Make sure the file exists.")

def main():
    # 1. Configure argparse to handle command-line arguments
    # This allows changing hyperparameters without modifying the code manually.
    parser = argparse.ArgumentParser(description='Train KAN-based PINN for Burgers Equation')
    
    parser.add_argument('--adam', type=int, default=1000, 
                        help='Number of epochs for Adam optimizer (Phase 1: Exploration & Grid Adapt)')
    parser.add_argument('--lbfgs', type=int, default=4000, 
                        help='Max iterations for LBFGS optimizer (Phase 2: Fine-tuning)')
    parser.add_argument('--name', type=str, default='experiment', 
                        help='Unique name tag for saving output files (models, plots, logs)')
    
    args = parser.parse_args()

    print(f"--- Starting Training Session: {args.name} ---")
    print(f"    Configuration: Adam={args.adam} epochs | LBFGS={args.lbfgs} iterations")

    # 2. Initialize the PINN model
    # We pass the epoch counts dynamically based on user input
    pinn = PINN(adam_epochs=args.adam, lbfgs_epochs=args.lbfgs)

    # 3. Execute Training
    # The train method returns the loss history list and the total execution time
    loss_history, train_time = pinn.train()

    # 4. Validation
    # Compute the final Relative L2 Error against the ground truth solution
    try:
        l2_error = pinn.compute_l2_error()
        print(f"Final L2 Error: {l2_error:.5e}")
    except Exception as e:
        print(f"Warning: Could not compute L2 error (Data missing?): {e}")
        l2_error = -1.0 # Placeholder for logs if calculation fails

    # 5. Save Artifacts
    # Save numeric metrics to a global summary text file (vital for tracking HPC jobs)
    pinn.save_metrics(l2_error, train_time, run_name=args.name)
    
    # Save the full model state dictionary and loss history for later analysis
    pinn.save_model(loss_history=loss_history, name=f"{args.name}_model.pth")

    print("Training finished successfully. Artifacts saved.")

if __name__ == "__main__":
    main()
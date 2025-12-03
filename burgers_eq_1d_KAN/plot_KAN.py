"""Visualization script for PINN predictions and training metrics.

This script loads a trained PINN model and generates plots showing:
  - Full space-time prediction contour
  - Solution slices at specific times vs exact solution
  - Training loss evolution (if loss history available)

Usage:
    python plot_KAN.py
"""

from pinn_KAN import PINN
import numpy as np
import os


def main():
    """Load model and generate visualization plots."""
    # Initialize PINN (will load data from disk)
    pinn = PINN()

    # Load pre-trained model weights
    pinn.load_model()

    # Plot predicted solution vs exact solution
    pinn.plot_solution()

    # Plot training loss history if available
    loss_path = os.path.join("saved_models", "loss_history_KAN.npy")
    if os.path.exists(loss_path):
        loss_history = np.load(loss_path)
        pinn.plot_loss_history(loss_history)
    else:
        print("\nℹ  No loss history found. Run train_KAN.py first to generate it.")

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
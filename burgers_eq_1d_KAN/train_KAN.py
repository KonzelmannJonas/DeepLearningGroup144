"""Training script for PINN with KAN.

This script trains the Physics-Informed Neural Network using KAN architecture
to solve the 1D Burgers' equation. The trained model and loss history are saved.

Usage:
    python train_KAN.py
"""

from pinn_KAN import PINN


def main():
    """Train the PINN model and save results."""
    # Initialize PINN with network, domain, and data
    pinn = PINN()

    # Train with two phases: Adam optimizer + LBFGS optimizer
    loss_history = pinn.train()

    # Save model weights and training history
    pinn.save_model()

    print("\n✓ Training complete!")
    print("✓ Model saved to ./saved_models/pinn_model_KAN.pth")
    print("✓ Loss history saved to ./saved_models/loss_history_KAN.npy")


if __name__ == "__main__":
    main()
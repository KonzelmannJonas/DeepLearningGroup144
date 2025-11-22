from pinn_KAN import PINN
import numpy as np
import os

def main():
    pinn = PINN()
    pinn.load_model()
    pinn.plot_solution()

    # Only load the KAN-specific loss history file
    loss_path = os.path.join('saved_models', 'loss_history_KAN.npy')
    if os.path.exists(loss_path):
        loss_history = np.load(loss_path)
        pinn.plot_loss_history(loss_history)
    else:
        print(f"No KAN loss history found at {loss_path}. Run `train_KAN.py` to generate it.")
    
if __name__ == "__main__":
    main()
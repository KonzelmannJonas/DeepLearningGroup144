from pinn import PINN
import numpy as np
import os

def main():
    pinn = PINN()
    loss_history = pinn.train()
    pinn.save_model()

    os.makedirs('./saved_models', exist_ok=True)
    np.save('./saved_models/loss_history.npy', np.array(loss_history))
    print("Loss history saved to ./saved_models/loss_history.npy")
    
if __name__ == "__main__":
    main()
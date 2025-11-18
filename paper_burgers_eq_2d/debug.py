from pinn_model import PINN
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_solution(xx, yy, a):
    plt.contourf(xx, yy, a, levels=50, cmap='jet')
    plt.show()

def main():
    pinn = PINN()
    pinn.load_model()
    
    # Prediction grid
    N_x = 256
    x = np.linspace(0, 1, N_x)
    y = np.linspace(0, 1, N_x)

    # Create meshgrid
    xx, yy = np.meshgrid(x, y)
    xy_flat = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Select time slices to plot
    time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    ts = 1
    x_input = torch.tensor(np.hstack((np.full((xy_flat.shape[0], 1), ts),xy_flat)), dtype=torch.float32).to(pinn.device)
    print("x_input:", x_input)
    
    Y_pred = pinn.predict(x_input).cpu().numpy()
    u = Y_pred[:, 0].reshape(N_x, N_x)
    v = Y_pred[:, 1].reshape(N_x, N_x)
    
    D = pinn.network.boundary_distance(x_input)
    Y_p = pinn.network.boundary_solution(x_input)

    print(f"boundary distance D at t={ts}:", D)
    print(f"boundary solution Y_p at t={ts}:", Y_p)
    
    plot_solution(xx, yy, u)
    plot_solution(xx, yy, v)

if __name__ == "__main__":
    main()
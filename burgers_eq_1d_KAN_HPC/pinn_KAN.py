"""
Physics-Informed Neural Network (PINN) with Kolmogorov-Arnold Networks (KAN) adaptation to HPC.

This module implements a KAN-based PINN to solve the 1D Viscous Burgers' equation:
    u_t + u * u_x - nu * u_xx = 0

Key Features:
- Architecture: KAN [2, 5, 5, 5, 1] with cubic B-splines (k=3) and grid size G=5.
- Training: Hybrid optimization strategy (Adam for exploration + LBFGS for high-precision fine-tuning).
- Adaptation: Dynamic grid updates based on activation distribution during the initial training phase.
- Integration: Ready for High-Performance Computing (HPC) workflows via argparse.

Author: Group 144 / [Your Name]
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import time
import os
import argparse  # ADDED: Necessary to read arguments from the terminal (HPC)
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec
from kan_lib.KANLayer import KANLayer
    
class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Architecture.
    
    Replaces standard MLPs with learnable B-spline activation functions on edges.
    """
    def __init__(self):
        super().__init__()
        # Architecture: [Input, Hidden, Hidden, Hidden, Output]
        # Input (2): (x, t)
        # Output (1): u(x,t)
        layers = [2, 5, 5, 5, 1]

        # KAN-specific parameters
        grid_size = 5  # Defines the spline resolution (G).
        k = 3          # Polynomial order (cubic B-splines).

        self.layers = nn.ModuleList()

        # Create the layers dynamically
        for i in range(len(layers) - 1):
            in_dim = layers[i]
            out_dim = layers[i + 1]

            # Instantiate the KANLayer
            # Pass 'num' (grid size) and 'k' (spline order)
            self.layers.append(
                KANLayer(in_dim, out_dim, num=grid_size, k=k)
            )

    def forward(self, x):
        # The loop is simpler now:
        for layer in self.layers:
            # KANLayer returns 4 values; we only want the first (x), which is the transformed activation.
            # DO NOT apply tanh here — the layer itself contains the non-linearity (splines).
            x, _, _, _ = layer(x)
            
        return x

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) wrapper for the KAN model.
    Encapsulates the physics (residuals), data loading, and training loop.
    """
    # MODIFICATION: Add arguments to the constructor to control epochs externally
    def __init__(self, adam_epochs=2000, lbfgs_epochs=3000):
        super(PINN, self).__init__()
        
        # ADDED: Store epochs as class variables to use in train()
        self.adam_epochs = adam_epochs
        self.lbfgs_epochs = lbfgs_epochs

        # set up neural network parameters
        self.network = KAN()
        
        # MODIFICATION: Real GPU detection. If available, use 'cuda', otherwise 'cpu'.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Initializing PINN on device: {self.device}") # ADDED: Log to verify where it runs
        
        # MODIFICATION: Move the network to the GPU immediately
        self.network.to(self.device)
        
        self.optimizer_adam = torch.optim.Adam(self.network.parameters(), lr=0.01) # Higher LR for KAN
        # self.optimizer = torch.optim.LBFGS... (Initialized later in train phase 2)

        # set up experiment parameters
        torch.set_default_dtype(torch.float32)
        # spatial and temporal domain boundaries
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

        #viscosity coefficient
        self.nu = 0.01 / np.pi

        # collocation points
        self.N_f = 10000
        self.N_0 = 200
        self.N_b = 200

        # Initial and boundary conditions
        X_f = np.random.rand(self.N_f, 2)
        X_f[:, 0] = X_f[:, 0] * (self.x_max - self.x_min) + self.x_min  # x in [-1, 1]
        X_f[:, 1] = X_f[:, 1] * (self.t_max - self.t_min) + self.t_min    # t in [0, 1]

        # Initial condition on velocity: u(x, 0) = -sin(pi * x)
        x0 = np.linspace(self.x_min, self.x_max, self.N_0)[:, None]
        t0 = np.zeros_like(x0)
        u0 = -np.sin(np.pi * x0)

        # Boundary conditions: u(-1, t) = 0, u(1, t) = 0
        tb = np.linspace(self.t_min, self.t_max, self.N_b)[:, None]
        xb_left = np.ones_like(tb) * self.x_min
        xb_right = np.ones_like(tb) * self.x_max
        ub_left = np.zeros_like(tb)
        ub_right = np.zeros_like(tb)

        # Convert to PyTorch tensors
        # MODIFICATION: Add .to(self.device) to ALL tensors. 
        # If network is on GPU and data on CPU, code will fail.
        self.X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(self.device)
        self.x0 = torch.tensor(x0, dtype=torch.float32).to(self.device)
        self.t0 = torch.tensor(t0, dtype=torch.float32).to(self.device)
        self.u0 = torch.tensor(u0, dtype=torch.float32).to(self.device)
        self.tb = torch.tensor(tb, dtype=torch.float32).to(self.device)
        self.xb_left = torch.tensor(xb_left, dtype=torch.float32).to(self.device)
        self.xb_right = torch.tensor(xb_right, dtype=torch.float32).to(self.device)
        self.ub_left = torch.tensor(ub_left, dtype=torch.float32).to(self.device)
        self.ub_right = torch.tensor(ub_right, dtype=torch.float32).to(self.device)
        
        # load ground truth data
        # ADDED: Try-except block in case data loading fails, preventing script crash
        try:
            data = scipy.io.loadmat('./data/burgers_shock.mat')
            
            self.t = data['t'].flatten()[:,None]
            self.x = data['x'].flatten()[:,None]
            self.Exact = np.real(data['usol']).T

            X, T = np.meshgrid(self.x, self.t)

            X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
            u_star = self.Exact.flatten()[:,None]

            # convert to pytorch tensors
            # MODIFICATION: Also move validation data to GPU (.to(self.device))
            self.X_star = torch.tensor(X_star, dtype=torch.float32).to(self.device)
            self.u_star = torch.tensor(u_star, dtype=torch.float32).to(self.device)
        except Exception as e:
            print(f"Warning: Could not load ./data/burgers_shock.mat: {e}")
        
    def pde_residual(self, X):
        """
        Computes the physics residual (Burgers' Equation) using AutoGrad.
        Residual: f = u_t + u*u_x - nu*u_xx
        """
        x = X[:, 0:1] 
        t = X[:, 1:2] 
        u = self.network(torch.cat([x, t], dim=1)) # network output u(x,t)

        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        f = u_t + u * u_x - self.nu * u_xx  # Burgers' equation residual
        return f

    def loss_func(self):
        """
        Defines the total loss function: Physics Loss + IC Loss + BC Loss.
        """
        # PDE residual loss
        f_pred = self.pde_residual(self.X_f)
        loss_f = torch.mean(f_pred**2)

        # Initial condition loss
        u0_pred = self.network(torch.cat([self.x0, self.t0], dim=1))
        loss_0 = torch.mean((u0_pred - self.u0)**2)

        # Boundary condition loss
        u_left_pred = self.network(torch.cat([self.xb_left, self.tb], dim=1))
        u_right_pred = self.network(torch.cat([self.xb_right, self.tb], dim=1))
        loss_b = torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2)

        loss = loss_f + loss_0 + loss_b
        return loss

    def train(self):
        """
        Main training loop.
        Phase 1: Adam optimizer with grid adaptation.
        Phase 2: LBFGS optimizer for fine-tuning.
        """
        print("Starting training...")
        start_time = time.perf_counter()
        loss_history = []
        
        # --- PHASE 1: Adam Optimizer (Coarse tuning & Grid adaptation) ---
        # MODIFICATION: Use self.adam_epochs variable instead of fixed number
        print(f"--> Phase 1: Adam Optimizer ({self.adam_epochs} epochs)")
        
        for epoch in range(self.adam_epochs):
            # Grid update logic: adapt splines to the solution shape
            if epoch % 50 == 0 and epoch < self.adam_epochs // 2:
                with torch.no_grad():
                    x_input = self.X_f 
                    for layer in self.network.layers:
                        layer.update_grid_from_samples(x_input)
                        x_input, _, _, _ = layer(x_input) # Forward pass for next layer

            self.optimizer_adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_adam.step()
            
            loss_history.append(loss.item()) 

            if (epoch+1) % 100 == 0:
                print(f'[Adam] Epoch {epoch+1}/{self.adam_epochs}, Loss: {loss.item():.5e}')

        # --- PHASE 2: LBFGS Optimizer (Fine tuning) ---
        # MODIFICATION: Use self.lbfgs_epochs
        print(f"--> Phase 2: LBFGS Optimizer ({self.lbfgs_epochs} iterations)")
        
        self.optimizer = torch.optim.LBFGS(
            self.network.parameters(), 
            lr=1.0, 
            # MODIFICATION: Update max_iter with the passed parameter
            max_iter=self.lbfgs_epochs, 
            max_eval=self.lbfgs_epochs, 
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        # Closure function required by LBFGS for re-evaluating loss
        def closure():
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            loss_history.append(loss.item())
            return loss

        # Perform the optimization step (runs multiple iterations internally)
        self.optimizer.step(closure)
        
        print(f'[LBFGS] Final Loss: {loss_history[-1]:.5e}')

        total_time = time.perf_counter() - start_time
        print(f"Training complete! Total time: {total_time:.2f} seconds")
        
        # Attempt to save the model state
        try:
            os.makedirs('./saved_models', exist_ok=True)
            self.save_model(loss_history=loss_history)
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

        # MODIFICATION: Also return total time to save it later
        return loss_history, total_time
        
    def predict(self, X: torch.Tensor):
        self.network.eval()
        with torch.no_grad():
            # ADDED: Safety check. If X comes from outside and is on CPU, move to GPU.
            if X.device != self.device:
                X = X.to(self.device)
            u_pred = self.network(X)
        return u_pred
    
    def save_model(self, root="./saved_models", name="pinn_model_KAN.pth", loss_history=None):
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, name)
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")

        # Optionally save loss history if provided
        if loss_history is not None:
            try:
                np.save(os.path.join(root, 'loss_history_KAN.npy'), np.array(loss_history))
                print(f"Loss history saved to {os.path.join(root, 'loss_history_KAN.npy')}")
            except Exception as e:
                print(f"Warning: could not save loss history: {e}")

    def load_model(self, path="./saved_models/pinn_model_KAN.pth"):
        self.network.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    
    def compute_l2_error(self):
        """ Computes relative L2 error against ground truth. """
        u_pred = self.predict(self.X_star)
        
        # MODIFICATION: ADDED .cpu() before .numpy()
        # NumPy cannot work with tensors on GPU. If skipped, this raises an error.
        u_pred_np = u_pred.cpu().numpy()
        u_star_np = self.u_star.cpu().numpy()
        
        error_u = np.linalg.norm(u_star_np - u_pred_np, 2)/np.linalg.norm(u_star_np, 2)
        return error_u

    # ADDED: New function to save numeric results to a text file.
    # This is vital for comparing HPC experiments without opening images.
    def save_metrics(self, l2_error, total_time, run_name):
        os.makedirs('./results', exist_ok=True)
        file_path = './results/experiment_summary.txt'
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(file_path, "a") as f:
            f.write(f"Run: {run_name:<20} | Date: {timestamp} | "
                    f"Adam: {self.adam_epochs} | LBFGS: {self.lbfgs_epochs} | "
                    f"L2 Error: {l2_error:.5e} | Time: {total_time:.2f}s | Device: {self.device}\n")
        print(f"Metrics saved to {file_path}")

    def plot_solution(self, root="./saved_plots/", name="prediction_KAN.png"):
        """ Generates visualization plots for the solution at snapshots. """
        N_x, N_t = 256, 100
        x = np.linspace(self.x_min, self.x_max, N_x)
        t = np.linspace(self.t_min, self.t_max, N_t)
        X, T = np.meshgrid(x, t)
        XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        # MODIFICATION: Ensure grid tensor also goes to GPU
        XT_tensor = torch.tensor(XT, dtype=torch.float32).to(self.device)

        # MODIFICATION: .cpu() necessary to move prediction from GPU to RAM for plotting
        u_pred = self.predict(XT_tensor).cpu().numpy().reshape(N_t, N_x)

        time_slices = [0.0, 0.25, 0.5, 0.75]

        # Create figure and GridSpec
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(
            3,                         # 3 rows
            len(time_slices),          # same number of columns as time_slices
            height_ratios=[2, 0.1, 1], # top contour, small spacer, bottom plots
            hspace=0.4,                # vertical spacing
            wspace=0.3                 # horizontal spacing
        )

        # Top: contour spanning all columns
        ax_contour = fig.add_subplot(gs[0, :])
        contour = ax_contour.contourf(T.T, X.T, u_pred.T, levels=100, cmap='coolwarm')
        fig.colorbar(contour, ax=ax_contour, label='u(x,t)')
        ax_contour.set_xlabel('t')
        ax_contour.set_ylabel('x')
        ax_contour.set_title("Predicted solution u(x,t) via PINN")

        # Bottom: one axis per time slice, sharing the same columns
        for i, t_slice in enumerate(time_slices):
            ax_1d = fig.add_subplot(gs[2, i])

            t_idx = int(t_slice * (N_t - 1))
            u_pred_slice = u_pred[t_idx, :]

            # ADDED: Check if exact data is missing (avoids error if loading failed)
            if hasattr(self, 'Exact'):
                ax_1d.plot(self.x, self.Exact[t_idx, :], 'r--', label='Exact', linewidth=2)
            
            ax_1d.plot(x, u_pred_slice, 'b-', label="Predicted", linewidth=2)
            ax_1d.set_xlabel('x')
            ax_1d.set_ylabel('u')
            ax_1d.set_title(f't = {t_slice}')
            ax_1d.grid(True, alpha=0.3)
            ax_1d.legend()

        os.makedirs(root, exist_ok=True)
        fig.savefig(os.path.join(root, name), dpi=300, bbox_inches='tight')
        # plt.show() # MODIFICATION: Commented out because HPC has no screen and might error
        plt.close(fig)

    def plot_loss_history(self, loss_history, root="./saved_plots/", name="loss_history_KAN.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Total Loss', color='blue', linewidth=2)
        
        plt.yscale('log') 
        
        plt.title('Evolution of Loss during Training', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (Log Scale)', fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, name)
        plt.savefig(path)
        print(f"Graphic saved {path}")
        # plt.show() # MODIFICATION: Commented out for the same reason (HPC without screen)
        plt.close()

# ==========================================
# MAIN EXECUTION BLOCK (HPC STRUCTURE)
# ==========================================
if __name__ == "__main__":
    # ADDED: Configure 'argparse'.
    # This allows your 'submit.sh' script to pass variables like:
    # python main.py --adam 2000 --lbfgs 3000 --name "Test1"
    parser = argparse.ArgumentParser(description='Run KAN-PINN on HPC')
    parser.add_argument('--adam', type=int, default=2000, help='Epochs for Adam optimizer')
    parser.add_argument('--lbfgs', type=int, default=3000, help='Max iterations for LBFGS optimizer')
    parser.add_argument('--name', type=str, default='experiment', help='Name of the run for file saving')
    
    args = parser.parse_args()
    
    print("========================================")
    print(f"  RUNNING EXPERIMENT: {args.name}")
    print(f"  ADAM Epochs: {args.adam}")
    print(f"  LBFGS Epochs: {args.lbfgs}")
    print("========================================")

    # ADDED: Instantiate model passing arguments received from terminal
    model = PINN(adam_epochs=args.adam, lbfgs_epochs=args.lbfgs)
    
    # 3. Train (Return time as well)
    loss_history, train_time = model.train()
    
    # 4. Compute Metrics
    l2_error = model.compute_l2_error()
    print(f"Final L2 Error: {l2_error:.5e}")
    
    # 5. Save Results (Call new function to save to text file)
    model.save_metrics(l2_error, train_time, run_name=args.name)
    model.save_model(loss_history=loss_history, name=f"{args.name}_model.pth")
    
    # Attempt to plot (protected by try-except in case graphics backend fails)
    try:
        model.plot_solution(name=f"{args.name}_prediction.png")
        model.plot_loss_history(loss_history, name=f"{args.name}_loss.png")
    except Exception as e:
        print(f"Skipping plotting due to error (likely no display): {e}")
    
    print("Done.")
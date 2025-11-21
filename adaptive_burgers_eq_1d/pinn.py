import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
import time
import os

# Set seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================
# 1. Network Architectures
# ==========================================

class FNN(nn.Module):
    """
    Standard Feed Forward Network with Tanh activation.
    Flexible layer sizes passed via constructor.
    """
    def __init__(self, layer_sizes):
        super().__init__()
        self.activation = nn.Tanh() # as it naturally rescales to [-1, 1]
        self.layers = nn.ModuleList()

        # Construct linear layers dynamically
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            # Xavier/Glorot Normal Initialization (common for Tanh)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# class KAN(nn.Module):
#     def __init__(self, ...):
#         ...
#     def forward(self, x):
#         ...


# ==========================================
# 2. PINN Solver Class
# ==========================================

class PINN_Burgers:
    def __init__(self, network, device):
        """
        I put network as an input so that we can flexibly change architecture of the PINN
        To be coherent with that, also device is in input (otherwise error may be risen) 
        """
        self.device = device
        self.network = network.to(device)
        self.num_epochs = 5000 

        # Physical parameters
        self.nu = 0.01 / np.pi
        
        # Domain boundaries
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

        # Load ground truth data (Wrap in try-except to avoid crash if file missing)
        try:
            data = scipy.io.loadmat('./data/burgers_shock.mat')
            self.t = data['t'].flatten()[:,None]
            self.x = data['x'].flatten()[:,None]
            self.Exact = np.real(data['usol']).T
            
            X, T = np.meshgrid(self.x, self.t)
            X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
            u_star = self.Exact.flatten()[:,None]
            
            # convert to pytorch tensors
            self.X_star = torch.tensor(X_star, dtype=torch.float32).to(self.device)
            self.u_star = torch.tensor(u_star, dtype=torch.float32).to(self.device)
            self.has_ground_truth = True
        except FileNotFoundError:
            print("Warning: Ground truth data './data/burgers_shock.mat' not found. L2 error computation will be skipped.")
            self.has_ground_truth = False

        # Initialize data containers
        self.X_f = None # Collocation points
        self.X_b = None # Boundary points
        self.X_0 = None # Initial condition points
        self.U_0 = None # Initial condition values
        
        # Generate Data
        self.init_training_data()
        
    def init_training_data(self, N_f=2000, N_0=100, N_b=100):
        """Generates initial training data for X_f, X_b and X_0"""
        
        # Collocation Points
        X_f_np = np.random.rand(N_f, 2)
        X_f_np[:, 0] = X_f_np[:, 0] * (self.x_max - self.x_min) + self.x_min # x
        X_f_np[:, 1] = X_f_np[:, 1] * (self.t_max - self.t_min) + self.t_min # t
        
        self.X_f = torch.tensor(X_f_np, dtype=torch.float32, requires_grad=True).to(self.device)

        # Initial Condition u(x,0) = -sin(pi*x)
        x0 = np.linspace(self.x_min, self.x_max, N_0)[:, None]
        t0 = np.zeros_like(x0)
        u0 = -np.sin(np.pi * x0)
        
        self.X_0 = torch.tensor(np.hstack((x0, t0)), dtype=torch.float32).to(self.device)
        self.U_0 = torch.tensor(u0, dtype=torch.float32).to(self.device)

        # Boundary Conditions u(-1, t) = 0, u(1, t) = 0
        tb = np.linspace(self.t_min, self.t_max, N_b)[:, None]
        
        # Left boundary (-1, t) and Right boundary (1, t)
        X_bl = np.hstack((np.ones_like(tb) * self.x_min, tb))
        X_br = np.hstack((np.ones_like(tb) * self.x_max, tb))
        
        self.X_b = torch.tensor(np.vstack((X_bl, X_br)), dtype=torch.float32).to(self.device)
        self.U_b = torch.zeros((X_bl.shape[0] + X_br.shape[0], 1), dtype=torch.float32).to(self.device)

    def pde_residual(self, X):
        # Clone X to ensure we can compute gradients relative to input
        if not X.requires_grad:
            X = X.clone()
            X.requires_grad = True

        # simpler alternative but may be problematic
        # if X.requires_grad==False
        #    X.requires_grad = True
        
        u = self.network(X)
        
        # First derivatives: I derive wrt X=[x, t], then I separate u_x, u_t (less effort)
        grads = autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        
        # Second derivative (u_xx)
        grads_x = autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = grads_x[:, 0:1]
        
        # Burgers Equation Residual
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_function(self):
        
        # 1. Physics Loss
        f_pred = self.pde_residual(self.X_f)
        loss_f = torch.mean(f_pred ** 2)
        
        # 2. Initial Condition Loss
        u0_pred = self.network(self.X_0)
        loss_0 = torch.mean((u0_pred - self.U_0) ** 2)
        
        # 3. Boundary Condition Loss
        ub_pred = self.network(self.X_b)
        loss_b = torch.mean((ub_pred - self.U_b) ** 2)
        
        return loss_f + loss_0 + loss_b
    
    def train(self):
        """
        Runs standard training using only the initial, fixed set of collocation points (X_f).
        """
        print("Starting Standard Training (Non-Adaptive)...")
        start_time = time.perf_counter()
        
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.network.train()
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}')

        total_time = time.perf_counter() - start_time
        print(f"Standard Training complete! Total time: {total_time:.2f} seconds")

    def RAR_D_adaptive_sampling(self, k=2, c=0, N_cand=10000, num_add=10):
        """
        Residual-based Adaptive Refinement (RAR-D).
        
        From the paper the results show that the best accuracy is achieved for K=2, c=0
        Since the purpose of our work is not to investigate the parameters of adaptive sampling,
        we could let them fixed and not as inputs. 
        """
        # Candidate points
        X_cand = np.random.rand(N_cand, 2)
        X_cand[:, 0] = X_cand[:, 0] * (self.x_max - self.x_min) + self.x_min
        X_cand[:, 1] = X_cand[:, 1] * (self.t_max - self.t_min) + self.t_min
        
        X_cand = torch.tensor(X_cand, dtype=torch.float32).to(self.device)
        X_cand.requires_grad = True
        
        # Evaluate Residuals (detach because no grad needed for p_distribution)
        f_cand = self.pde_residual(X_cand)
        f_cand_val = torch.abs(f_cand).detach().cpu().numpy().flatten()
        
        ## RAR_D core
        # Calculate Error Distribution (PDF) ...
        mean_val = np.mean(np.power(f_cand_val, k))
        err_eq = (np.power(f_cand_val, k) / mean_val) + c
        p_distribution = err_eq / np.sum(err_eq)
        
        # ... sample points based on distribution ...
        indices = np.random.choice(N_cand, size=num_add, replace=False, p=p_distribution)
        X_add = X_cand[indices]
        
        # ... and add to training set (with DETACH - this "breaks" computations of gradient).
        # This is needed bcs in adaptive sampling we restart the training at every step, just
        #  like creating a new network every time we resample.
        # By using detach we break the connection between different "iterations", so that 
        # it computes gradient just for the current network (current sampling).
        # Detach ==> prevent infinite graph growth.        
        new_X_f = torch.cat((self.X_f.detach(), X_add), dim=0)
        self.X_f = new_X_f
        self.X_f.requires_grad = True
        
        avg_resid = np.mean(f_cand_val)
        print(f"[RAR] Added {num_add} points. Total Collocation: {len(self.X_f)}. Avg Residual on candidates: {avg_resid:.5e}")

    def train_RAR(self, rar_iter=5):
        print(f"Starting training on {self.device}...")
        start_time = time.perf_counter()

        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        total_loops = rar_iter + 1 
        
        for loop in range(total_loops):
            is_rar_step = loop > 0 
            
            # Adaptive Sampling Step
            if is_rar_step:
                print(f"\n=== RAR Cycle {loop}/{rar_iter}: Adding Points ===")
                self.RAR_D_adaptive_sampling() 

            # Training Step
            self.network.train()
            print(f"--- Cycle {loop}: Training ... ---")
            
            for epoch in range(self.num_epochs):
                optimizer.zero_grad()
                loss = self.loss_function()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 1000 == 0:
                    # Fix NameError: uses self.num_epochs
                    print(f'Cycle {loop}, Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}')
            
            print(f"Cycle {loop} completed. Loss: {loss.item():.5e}")

        total_time = time.perf_counter() - start_time
        print(f"Training complete! Total time: {total_time:.2f} seconds")
    
    def predict(self, X):
        self.network.eval()
        with torch.no_grad():
            u = self.network(X)
        return u.cpu().numpy()

    def compute_l2_error(self):
        if not hasattr(self, 'has_ground_truth') or not self.has_ground_truth:
            return 0.0
        # Ensure tensors are on CPU before converting to numpy for norm calculation
        u_pred = self.predict(self.X_star)
        error_u = np.linalg.norm(self.u_star.cpu().numpy()-u_pred,2)/np.linalg.norm(self.u_star.cpu().numpy(),2)
        return error_u

    def plot_solution(self, overlay_collocation_points=True, root="./saved_plots/", name="prediction.png"):
        N_x, N_t = 256, 100
        x = np.linspace(self.x_min, self.x_max, N_x)
        t = np.linspace(self.t_min, self.t_max, N_t)
        X, T = np.meshgrid(x, t)
        XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        XT_tensor = torch.tensor(XT, dtype=torch.float32).to(self.device)

        u_pred = self.predict(XT_tensor).reshape(N_t, N_x)

        time_slices = [0.0, 0.25, 0.5, 0.75]

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, len(time_slices), height_ratios=[2, 0.1, 1], hspace=0.4, wspace=0.3)

        # Top: contour
        ax_contour = fig.add_subplot(gs[0, :])
        contour = ax_contour.contourf(T, X, u_pred, levels=100, cmap='coolwarm')
        fig.colorbar(contour, ax=ax_contour, label='u(x,t)')
        ax_contour.set_xlabel('t')
        ax_contour.set_ylabel('x')
        ax_contour.set_title("Predicted solution u(x,t) via PINN")
        
        # overlay collocation points
        if overlay_collocation_points:
            Xf_np = self.X_f.detach().cpu().numpy()
            # Note: t on x-axis, x on y-axis for the contour plot convention
            ax_contour.scatter(Xf_np[:, 1], Xf_np[:, 0], marker='x', s=10, c='k', alpha=0.5, label='Collocation points')
            ax_contour.legend(loc='upper right')

        # Bottom: slices
        if hasattr(self, 'Exact'):
            for i, t_slice in enumerate(time_slices):
                ax_1d = fig.add_subplot(gs[2, i])
                t_idx = int(t_slice * (N_t - 1))
                ax_1d.plot(self.x, self.Exact[t_idx, :], 'r--', label='Exact', linewidth=2)
                ax_1d.plot(x, u_pred[t_idx, :], 'b-', label="Predicted", linewidth=2)
                ax_1d.set_title(f't = {t_slice}')
                ax_1d.set_xlabel('x')
                ax_1d.set_ylabel('u')
                ax_1d.grid(True, alpha=0.3)
                ax_1d.legend()

        os.makedirs(root, exist_ok=True)
        fig.savefig(os.path.join(root, name), dpi=100, bbox_inches='tight')
        plt.show()
        plt.close(fig)
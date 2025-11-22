import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec
from kan_lib.KANLayer import KANLayer
    
class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture: [Input, Hidden, Hidden, Hidden, Output]
        layers = [2, 5, 5, 5, 1]

        # KAN-specific parameters
        grid_size = 10  # Defines the spline resolution.
        k = 3          # Polynomial order (cubic)

        self.layers = nn.ModuleList()

        # Create the layers
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
            # KANLayer returns 4 values; we only want the first (x)
            # DO NOT apply tanh here — the layer is already non-linear.
            x, _, _, _ = layer(x)
            
        return x

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # set up neural network parameters
        self.network = KAN()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.network.to(self.device)
        self.optimizer_adam = torch.optim.Adam(self.network.parameters(), lr=0.01) # Higher LR for KAN
        self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=1.0)
        self.num_epochs = 2000

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
        self.X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True) # enable gradients for collocation points
        self.x0 = torch.tensor(x0, dtype=torch.float32)
        self.t0 = torch.tensor(t0, dtype=torch.float32)
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.tb = torch.tensor(tb, dtype=torch.float32)
        self.xb_left = torch.tensor(xb_left, dtype=torch.float32)
        self.xb_right = torch.tensor(xb_right, dtype=torch.float32)
        self.ub_left = torch.tensor(ub_left, dtype=torch.float32)
        self.ub_right = torch.tensor(ub_right, dtype=torch.float32)
        
        # load ground truth data
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
        
    def pde_residual(self, X):
        x = X[:, 0:1] 
        t = X[:, 1:2] 
        u = self.network(torch.cat([x, t], dim=1)) # network output u(x,t)

        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        f = u_t + u * u_x - self.nu * u_xx  # Burgers' equation residual
        return f

    def loss_func(self):

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


    # def train(self):
    #     print("Starting training...")
    #     start_time = time.perf_counter()
    #     loss_history = []  # <--- Create an empty list to store loss values
            
    #     for epoch in range(self.num_epochs):

    #         # Bloc d'Actualització del Grid
    #         if epoch % 50 == 0 and epoch < self.num_epochs // 2:
                
    #             # 1. Comencem amb l'entrada original del problema (x, t)
    #             # Mida: [10000, 2]
    #             x_input = self.X_f 
                
    #             # 2. Recorrem les capes una per una
    #             for layer in self.network.layers:
                    
    #                 # A. Actualitzem el grid d'AQUESTA capa amb l'entrada que té ara mateix.
    #                 # Si és la primera capa, x_input té mida 2.
    #                 # Si és la segona, x_input tindrà mida 10.
    #                 layer.update_grid_from_samples(x_input)
                    
    #                 # B. "empenyem" les dades a través de la capa per preparar-les per la següent.
    #                 # layer(x_input) retorna 4 coses, només volem la primera (la sortida transformada)
    #                 # Ara x_input passa de ser l'entrada d'aquesta capa a ser la sortida.
    #                 x_input, _, _, _ = layer(x_input)
                    
    #                 # En la següent volta del bucle, 'x_input' ja tindrà la mida correcta (10)
    #                 # per a la següent capa.
                     
    #         self.optimizer.zero_grad()
    #         loss = self.loss_func()
    #         loss.backward()
    #         self.optimizer.step()
            
    #         # Saving the loss value to the history list
    #         loss_history.append(loss.item()) 

    #         if (epoch+1) % 50 == 0:
    #             print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}')

    #     total_time = time.perf_counter() - start_time
    #     print(f"Training complete! Total time: {total_time:.2f} seconds")
    #     # Save model and loss history to disk (if desired)
    #     try:
    #         # Ensure saved_models directory exists
    #         os.makedirs('./saved_models', exist_ok=True)
    #         # Save the network weights and loss history using save_model
    #         self.save_model(loss_history=loss_history)
    #     except Exception:
    #         # If saving fails here, still return the history so caller can handle it
    #         pass

    #     return loss_history    

    def train(self):
        print("Starting training...")
        start_time = time.perf_counter()
        loss_history = []
        
        # --- PHASE 1: Adam Optimizer (Coarse tuning & Grid adaptation) ---
        print("--> Phase 1: Adam Optimizer")
        # optimizer_adam = torch.optim.Adam(self.network.parameters(), lr=0.01)
        adam_epochs = 1000  # Half of the training for adaptation
        
        for epoch in range(adam_epochs):
            # Grid update logic: adapt splines to the solution shape
            # Only done during the first half to avoid instability
            if epoch % 50 == 0 and epoch < adam_epochs // 2:
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
                print(f'[Adam] Epoch {epoch+1}/{adam_epochs}, Loss: {loss.item():.5e}')

        # --- PHASE 2: LBFGS Optimizer (Fine tuning) ---
        print("--> Phase 2: LBFGS Optimizer")
        # LBFGS needs a high LR (1.0) to estimate curvature effectively
        self.optimizer = torch.optim.LBFGS(
            self.network.parameters(), 
            lr=1.0, 
            max_iter=2000, 
            max_eval=2000, 
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

        return loss_history
        
    def predict(self, X: torch.Tensor):
        self.network.eval()
        with torch.no_grad():
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
        u_pred = self.predict(self.X_star)
        error_u = np.linalg.norm(self.u_star-u_pred,2)/np.linalg.norm(self.u_star,2)
        return error_u

    def plot_solution(self, root="./saved_plots/", name="prediction_KAN.png"):
        N_x, N_t = 256, 100
        x = np.linspace(self.x_min, self.x_max, N_x)
        t = np.linspace(self.t_min, self.t_max, N_t)
        X, T = np.meshgrid(x, t)
        XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        XT_tensor = torch.tensor(XT, dtype=torch.float32).to(self.device)

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

            ax_1d.plot(self.x, self.Exact[t_idx, :], 'r--', label='Exact', linewidth=2)
            ax_1d.plot(x, u_pred_slice, 'b-', label="Predicted", linewidth=2)
            ax_1d.set_xlabel('x')
            ax_1d.set_ylabel('u')
            ax_1d.set_title(f't = {t_slice}')
            ax_1d.grid(True, alpha=0.3)
            ax_1d.legend()

        os.makedirs(root, exist_ok=True)
        fig.savefig(os.path.join(root, name), dpi=300, bbox_inches='tight')
        plt.show()
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
        plt.show()
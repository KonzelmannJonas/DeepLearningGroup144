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
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh() # naturally scales network output to [-1, 1]
        layers = [2, 50, 50, 50, 50, 1]
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
        self.layers = nn.ModuleList(modules)   

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)

class PINN(nn.Module):
    def __init__(self, loss_weights=None, epochs=2000,use_data_loss=True, data_weight=1.0):
        super(PINN, self).__init__()
        # set up neural network parameters
        self.network = MLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.num_epochs = epochs
        #Loss weights
        # Initialize weights
        self.loss_weights = loss_weights or {'r': 1, 'bc': 1, 'ic': 1}
        self.loss_history = []
        self.weight_history = []
        # Balancing parameters
        self.update_frequency = 100  # Update weights every 100 steps
        self.alpha = 0.7  # Moving average parameter

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
        
        # load ground truth data (use path relative to this file so cwd doesn't matter)
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'burgers_shock.mat')
        
        data = scipy.io.loadmat(data_path)

        # New parameters for data loss
        self.use_data_loss = use_data_loss
        self.data_weight = data_weight

        self.t = data['t'].flatten()[:, None]
        self.x = data['x'].flatten()[:, None]
        self.Exact = np.real(data['usol']).T


        X, T = np.meshgrid(self.x, self.t)

        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_star = self.Exact.flatten()[:,None]

        # convert to pytorch tensors
        self.X_star = torch.tensor(X_star, dtype=torch.float32).to(self.device)
        self.u_star = torch.tensor(u_star, dtype=torch.float32).to(self.device)
        # Sample data points for training (you can use all or sample subset)
        self.setup_training_data()
        
        # Update loss weights to include data component
        if loss_weights is None:
            self.loss_weights = {'r': 1, 'bc': 1, 'ic': 1, 'data': data_weight}
        else:
            self.loss_weights = loss_weights
            if 'data' not in self.loss_weights:
                self.loss_weights['data'] = data_weight

        #New workflow where component gradients are computed sparately
    def compute_component_gradients(self, loss_dict):
    #"""Compute gradient norms for each loss component"""
        grad_norms = {}
    
        for name, loss in loss_dict.items():
        # Clear previous gradients for this component
            self.optimizer.zero_grad()
        
        # Compute gradients for this specific loss
            gradients = torch.autograd.grad(
                loss, 
                list(self.network.parameters()),  # convert iterator to list of tensors
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
        
        # Filter out None gradients and compute norm
            valid_grads = [g for g in gradients if g is not None]
            if valid_grads:
                grad_norms[name] = torch.stack([g.detach().norm(2) for g in valid_grads]).norm(2).item()
            else:
                grad_norms[name] = 0.0
                
        return grad_norms
    
    def setup_training_data(self, data_fraction=0.3):
        """Setup training data points from the exact solution"""
        # Sample a fraction of the data points for training
        n_total = self.X_star.shape[0]
        n_train = int(n_total * data_fraction)
        
        # Randomly select training points
        indices = np.random.choice(n_total, n_train, replace=False)
        self.X_data = self.X_star[indices].clone().detach().requires_grad_(True)
        self.u_data = self.u_star[indices].clone().detach()
        
        print(f"Using {n_train}/{n_total} data points for training")

    
    def update_loss_weights(self, grad_norms):
        #"""Update weights using reciprocal approach - all components adapt"""
        new_weights = self.loss_weights.copy()
    
        eps = 1e-8
        max_ratio = 100.0
    
    # Calculate target ratios for ALL components
        grad_norms_list = [gn for gn in grad_norms.values() if gn > eps]
        if grad_norms_list:
            mean_grad_norm = np.mean(grad_norms_list)
        else:
            mean_grad_norm = 1.0
    
        for name, grad_norm in grad_norms.items():
            if grad_norm > eps:
            # All components adjust toward mean gradient norm
                target_ratio = mean_grad_norm / (grad_norm + eps)
                target_ratio = np.clip(target_ratio, 1/max_ratio, max_ratio)
            
                new_weights[name] = (self.alpha * self.loss_weights[name] + 
                               (1 - self.alpha) * target_ratio)
    
        self.loss_weights = new_weights

    #Alternative as the base behaves weird
    def update_loss_weights_balanced(self, grad_norms):
        """Updated to handle data loss component"""
        new_weights = self.loss_weights.copy()
    
        eps = 1e-8
        max_ratio = 20.0
    
        # Use geometric mean as reference
        grad_vals = [gn for gn in grad_norms.values() if gn > eps]
        if grad_vals:
            ref_norm = np.exp(np.mean(np.log(grad_vals)))
        else:
            ref_norm = 1.0
    
        # Update ALL weights with different adaptation rates
        for name, grad_norm in grad_norms.items():
            if grad_norm > eps:
                target_ratio = ref_norm / (grad_norm + eps)
                target_ratio = np.clip(target_ratio, 1/max_ratio, max_ratio)
            
                # Different adaptation rates for different components
                if name == 'bc':
                    alpha = 0.95  # Very slow adaptation for BC
                elif name == 'data':
                    alpha = 0.9   # Moderate adaptation for data
                else:
                    alpha = 0.8   # Faster adaptation for PDE and IC
                
                new_weights[name] = (alpha * self.loss_weights[name] + 
                                   (1 - alpha) * target_ratio)
    
        self.loss_weights = new_weights
    def compute_loss_components(self):
        #"""Compute individual loss components including data loss"""
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
        
        # Data loss (if using data)
        loss_data = torch.tensor(0.0)
        if self.use_data_loss and hasattr(self, 'X_data'):
            u_data_pred = self.network(self.X_data)
            loss_data = torch.mean((u_data_pred - self.u_data)**2)
    
        return {'r': loss_f, 'ic': loss_0, 'bc': loss_b, 'data': loss_data}
    

    def train_gradient_balanced(self):
        """Training with gradient norm balancing including data loss"""
        print("Starting gradient-balanced training...")
        start_time = time.perf_counter()
    
        for epoch in range(self.num_epochs):
            # Compute individual loss components
            loss_dict = self.compute_loss_components()
        
            # Update weights periodically based on gradient norms
            if epoch % self.update_frequency == 0:
                with torch.no_grad():
                    grad_norms = self.compute_component_gradients(loss_dict)
                    self.update_loss_weights_balanced(grad_norms)
                    self.weight_history.append({
                        'epoch': epoch,
                        'weights': self.loss_weights.copy(),
                        'grad_norms': grad_norms
                    })
        
            # Compute weighted total loss (including data)
            total_loss = (self.loss_weights['r'] * loss_dict['r'] + 
                         self.loss_weights['ic'] * loss_dict['ic'] + 
                         self.loss_weights['bc'] * loss_dict['bc'] + 
                         self.loss_weights['data'] * loss_dict['data'])
        
            # Standard optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging
            if (epoch+1) % 10 == 0:
                loss_info = {
                    'epoch': epoch+1,
                    'total': total_loss.item(),
                    'r': loss_dict['r'].item(),
                    'ic': loss_dict['ic'].item(),
                    'bc': loss_dict['bc'].item(),
                    'data': loss_dict['data'].item(),
                    'weights': self.loss_weights.copy()
                }
                self.loss_history.append(loss_info)
                print(f'Epoch {epoch+1}/{self.num_epochs}, '
                      f'Total Loss: {total_loss.item():.5e}, '
                      f'Weights - r: {self.loss_weights["r"]:.3f}, '
                      f'ic: {self.loss_weights["ic"]:.3f}, '
                      f'bc: {self.loss_weights["bc"]:.3f}, '
                      f'data: {self.loss_weights["data"]:.3f}')

        total_time = time.perf_counter() - start_time
        # Save training logs
        try:
            self.save_training_info()
        except Exception as e:
            print(f"Failed to save training info: {e}")

        print(f"Gradient-balanced training complete! Total time: {total_time:.2f} seconds")
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
    
    def loss_func_wb(self):
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
    
    def loss_func_DualDimer(self):
        #reference: https://arxiv.org/pdf/2005.00615
        #Not yet implemented
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
    
    def loss_func_percentage(self):
        #Reference: https://arxiv.org/pdf/2005.00615
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
        
        #Set new weights
        self.lambda_f = loss_f / (loss_0 + loss_b + loss_f)
        self.lambda_0 = loss_0 / (loss_0 + loss_b + loss_f)
        self.lambda_b = loss_b / (loss_0 + loss_b + loss_f)

        loss = self.lambda_f * loss_f + self.lambda_0 * loss_0 + self.lambda_b * loss_b
        return loss
    
    def train(self, method='gradient_balanced'):
    #"""Unified training method with different balancing strategies"""
        if method == 'standard':
            self._train_standard()
        elif method == 'percentage':
            self._train_percentage()
        elif method == 'gradient_balanced':
            self.train_gradient_balanced()
        else:
            raise ValueError(f"Unknown training method: {method}")

    def _train_standard(self):
    #"""Original training method"""
        print("Starting standard training...")
        start_time = time.perf_counter()
        
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()  # Your original loss function
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}')

        total_time = time.perf_counter() - start_time
        # Save training logs
        try:
            self.save_training_info()
        except Exception as e:
            print(f"Failed to save training info: {e}")

        print(f"Standard training complete! Total time: {total_time:.2f} seconds")

    def _train_percentage(self):
    #"""Percentage-based weighting training"""
        print("Starting percentage-based training...")
        start_time = time.perf_counter()
        
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func_percentage()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}')

        total_time = time.perf_counter() - start_time
        # Save training logs
        try:
            self.save_training_info()
        except Exception as e:
            print(f"Failed to save training info: {e}")

        print(f"Percentage-based training complete! Total time: {total_time:.2f} seconds")
        
        
    def predict(self, X: torch.Tensor):
        self.network.eval()
        with torch.no_grad():
            u_pred = self.network(X)
        return u_pred
    
    def save_model(self, root="./saved_models", name="pinn_model.pth"):
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, name)
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")

    def save_training_info(self, root="./saved_models", name="training_info.json"):
        """Save training losses, weight history and basic hyperparameters to a JSON file."""
        import json
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, name)

        info = {
            'loss_history': [],
            'weight_history': [],
            'loss_weights': {k: float(v) for k, v in self.loss_weights.items()},
            'num_epochs': int(self.num_epochs),
            'nu': float(self.nu),
            'use_data_loss': bool(self.use_data_loss),
            'data_weight': float(self.data_weight)
        }

        for entry in getattr(self, 'loss_history', []):
            e = entry.copy()
            e['epoch'] = int(e.get('epoch', 0))
            for k in ['total', 'r', 'ic', 'bc', 'data']:
                if k in e:
                    try:
                        e[k] = float(e[k])
                    except Exception:
                        e[k] = None
            if 'weights' in e:
                e['weights'] = {kk: float(v) for kk, v in e['weights'].items()}
            info['loss_history'].append(e)

        for entry in getattr(self, 'weight_history', []):
            w = {
                'epoch': int(entry.get('epoch', 0)),
                'weights': {kk: float(v) for kk, v in entry.get('weights', {}).items()},
                'grad_norms': {kk: float(v) for kk, v in entry.get('grad_norms', {}).items()}
            }
            info['weight_history'].append(w)

        with open(path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Training info saved to {path}")

    def load_model(self, path="./saved_models/pinn_model.pth"):
        self.network.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    
    def compute_l2_error(self):
        u_pred = self.predict(self.X_star)
        error_u = np.linalg.norm(self.u_star-u_pred,2)/np.linalg.norm(self.u_star,2)
        return error_u

    def plot_solution(self, root="./saved_plots/", name="prediction.png"):
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
        plt.close('all')
    #NEW PLOT
    def plot_weight_evolution(self, root="./saved_plots/", name="weight_evolution.png"):
    #"""Plot the evolution of loss weights during training"""
        if not self.weight_history:
            print("No weight history available. Run gradient-balanced training first.")
            return
        
        epochs = [entry['epoch'] for entry in self.weight_history]
        weights_r = [entry['weights']['r'] for entry in self.weight_history]
        weights_ic = [entry['weights']['ic'] for entry in self.weight_history]
        weights_bc = [entry['weights']['bc'] for entry in self.weight_history]
    
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, weights_r, 'r-', label='PDE Residual Weight', linewidth=2)
        plt.plot(epochs, weights_ic, 'g-', label='Initial Condition Weight', linewidth=2)
        plt.plot(epochs, weights_bc, 'b-', label='Boundary Condition Weight', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title('Evolution of Loss Weights (Gradient Norm Balancing)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Often helpful to use log scale for weights
    
        os.makedirs(root, exist_ok=True)
        plt.savefig(os.path.join(root, name), dpi=300, bbox_inches='tight')
        plt.close('all')
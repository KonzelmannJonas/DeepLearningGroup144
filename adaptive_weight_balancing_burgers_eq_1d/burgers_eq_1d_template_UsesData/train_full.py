import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec
import json
from datetime import datetime
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
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
    def __init__(self, loss_weights=None, epochs=5000, use_data_loss=True, data_weight=1.0, 
                 output_dir=None):
        super(PINN, self).__init__()
        
        # Create output directory with timestamp
        if output_dir is None:
            # Get current directory where script is running
            current_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(current_dir, f"pinn_results_{timestamp}")
        else:
            self.output_dir = output_dir
            
        # Create subdirectories
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        print(f"Output directory created: {self.output_dir}")
        print(f"  Plots: {self.plots_dir}")
        print(f"  Models: {self.models_dir}")
        print(f"  Logs: {self.logs_dir}")
        
        # Set up neural network parameters
        self.network = MLP()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.num_epochs = epochs
        
        # Loss weights
        self.loss_weights = loss_weights or {'r': 1, 'bc': 1, 'ic': 1}
        self.loss_history = []
        self.weight_history = []
        
        # Balancing parameters
        self.update_frequency = 100
        self.alpha = 0.7

        # Set up experiment parameters
        torch.set_default_dtype(torch.float32)
        
        # Spatial and temporal domain boundaries
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

        # Viscosity coefficient
        self.nu = 0.01 / np.pi

        # Collocation points - reduced for faster initial testing
        self.N_f = 2000
        self.N_0 = 100
        self.N_b = 100
        
        # Data loss parameters
        self.use_data_loss = use_data_loss
        self.data_weight = data_weight

        # Generate training data
        self.generate_training_data()
        
        # Load ground truth data with fallback
        self.load_ground_truth()
        
        # Setup training data
        self.setup_training_data(data_fraction=0.1)
        
        # Update loss weights to include data component
        if loss_weights is None:
            self.loss_weights = {'r': 1, 'bc': 1, 'ic': 1, 'data': data_weight}
        else:
            self.loss_weights = loss_weights
            if 'data' not in self.loss_weights:
                self.loss_weights['data'] = data_weight

    def generate_training_data(self):
        """Generate collocation points and initial/boundary conditions"""
        # Collocation points for PDE residual
        X_f = np.random.rand(self.N_f, 2)
        X_f[:, 0] = X_f[:, 0] * (self.x_max - self.x_min) + self.x_min
        X_f[:, 1] = X_f[:, 1] * (self.t_max - self.t_min) + self.t_min

        # Initial condition: u(x, 0) = -sin(pi * x)
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
        self.X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
        self.x0 = torch.tensor(x0, dtype=torch.float32)
        self.t0 = torch.tensor(t0, dtype=torch.float32)
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.tb = torch.tensor(tb, dtype=torch.float32)
        self.xb_left = torch.tensor(xb_left, dtype=torch.float32)
        self.xb_right = torch.tensor(xb_right, dtype=torch.float32)
        self.ub_left = torch.tensor(ub_left, dtype=torch.float32)
        self.ub_right = torch.tensor(ub_right, dtype=torch.float32)

    def load_ground_truth(self):
        """Load ground truth data for testing with fallback"""
        try:
            # Try multiple possible paths for the data file
            possible_paths = [
                'burgers_shock.mat',
                './data/burgers_shock.mat',
                '../data/burgers_shock.mat',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'burgers_shock.mat')
            ]
            
            data_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            
            if data_path:
                print(f"Loading ground truth from {data_path}")
                data = scipy.io.loadmat(data_path)
                self.t = data['t'].flatten()[:, None]
                self.x = data['x'].flatten()[:, None]
                self.Exact = np.real(data['usol']).T

                X, T = np.meshgrid(self.x, self.t)
                X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
                u_star = self.Exact.flatten()[:,None]

                # Convert to PyTorch tensors
                self.X_star = torch.tensor(X_star, dtype=torch.float32)
                self.u_star = torch.tensor(u_star, dtype=torch.float32)
            else:
                print("Warning: Ground truth data file not found. Creating synthetic data.")
                self.create_synthetic_data()
                
        except Exception as e:
            print(f"Error loading ground truth: {e}. Creating synthetic data.")
            self.create_synthetic_data()

    def create_synthetic_data(self):
        """Create synthetic data for testing when real data is unavailable"""
        print("Creating synthetic ground truth data...")
        N_x, N_t = 256, 100
        self.x = np.linspace(self.x_min, self.x_max, N_x)[:, None]
        self.t = np.linspace(self.t_min, self.t_max, N_t)[:, None]
        
        # Create a simple synthetic solution
        X, T = np.meshgrid(self.x.flatten(), self.t.flatten())
        # Simple wave solution for testing: u(x,t) = -sin(pi*(x - t))
        self.Exact = -np.sin(np.pi * (X - T)).reshape(N_t, N_x)
        
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_star = self.Exact.flatten()[:,None]
        
        self.X_star = torch.tensor(X_star, dtype=torch.float32)
        self.u_star = torch.tensor(u_star, dtype=torch.float32)

    def setup_training_data(self, data_fraction=0.1):
        """Setup training data points from the exact solution"""
        # Sample a fraction of the data points for training
        n_total = self.X_star.shape[0]
        n_train = int(n_total * data_fraction)
        
        # Randomly select training points
        indices = np.random.choice(n_total, n_train, replace=False)
        self.X_data = self.X_star[indices].clone().detach().requires_grad_(True)
        self.u_data = self.u_star[indices].clone().detach()
        
        print(f"Using {n_train}/{n_total} data points for training")

    def compute_component_gradients(self, loss_dict):
        """Compute gradient norms for each loss component"""
        grad_norms = {}
    
        for name, loss in loss_dict.items():
            # Clear previous gradients for this component
            self.optimizer.zero_grad()
        
            # Compute gradients for this specific loss
            gradients = torch.autograd.grad(
                loss, 
                list(self.network.parameters()),
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
        """Compute individual loss components including data loss"""
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

    def pde_residual(self, X):
        """Compute Burgers' equation residual"""
        x = X[:, 0:1] 
        t = X[:, 1:2] 
        u = self.network(torch.cat([x, t], dim=1))

        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                           create_graph=True, retain_graph=True)[0]
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                           create_graph=True, retain_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                            create_graph=True, retain_graph=True)[0]

        f = u_t + u * u_x - self.nu * u_xx
        return f

    def train_gradient_balanced(self):
        """Training with gradient norm balancing including data loss"""
        print("Starting gradient-balanced training...")
        print(f"Initial weights: {self.loss_weights}")
        start_time = time.perf_counter()
        
        # Create log file for detailed output
        log_file = os.path.join(self.logs_dir, "training_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"PINN Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epochs: {self.num_epochs}, Use Data Loss: {self.use_data_loss}\n")
            f.write(f"Initial weights: {self.loss_weights}\n")
            f.write("="*80 + "\n")
    
        for epoch in range(self.num_epochs):
            # Compute individual loss components
            loss_dict = self.compute_loss_components()
        
            # Update weights periodically based on gradient norms
            if epoch % self.update_frequency == 0:
                with torch.no_grad():
                    grad_norms = self.compute_component_gradients(loss_dict)
                    self.update_loss_weights_balanced(grad_norms)
                    
                    # Record weight history
                    self.weight_history.append({
                        'epoch': epoch,
                        'weights': self.loss_weights.copy(),
                        'grad_norms': grad_norms,
                        'losses': {k: v.item() for k, v in loss_dict.items()}
                    })
                    
                    # Log detailed info every 1000 epochs
                    if epoch % 1000 == 0:
                        log_msg = f"\nEpoch {epoch}:\n"
                        log_msg += f"  Losses: { {k: f'{v.item():.2e}' for k, v in loss_dict.items()} }\n"
                        log_msg += f"  Grad Norms: { {k: f'{v:.2e}' for k, v in grad_norms.items()} }\n"
                        log_msg += f"  Weights: {self.loss_weights}\n"
                        print(log_msg)
                        
                        # Also write to log file
                        with open(log_file, 'a') as f:
                            f.write(log_msg)
        
            # Compute weighted total loss (including data)
            total_loss = (self.loss_weights['r'] * loss_dict['r'] + 
                         self.loss_weights['ic'] * loss_dict['ic'] + 
                         self.loss_weights['bc'] * loss_dict['bc'] + 
                         self.loss_weights['data'] * loss_dict['data'])
        
            # Standard optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging every 500 epochs
            if (epoch+1) % 500 == 0:
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
                
                progress_msg = f'Epoch {epoch+1}/{self.num_epochs}, Total Loss: {total_loss.item():.5e}'
                print(progress_msg)
                
                # Write progress to log file
                with open(log_file, 'a') as f:
                    f.write(progress_msg + "\n")

        total_time = time.perf_counter() - start_time
        
        # Write final summary to log file
        with open(log_file, 'a') as f:
            f.write("="*80 + "\n")
            f.write(f"Training completed in {total_time:.2f} seconds\n")
            f.write(f"Final weights: {self.loss_weights}\n")
        
        print(f"Gradient-balanced training complete! Total time: {total_time:.2f} seconds")
        print(f"Final weights: {self.loss_weights}")
        print(f"Training log saved to: {log_file}")

    def train_standard(self):
        """Standard training with fixed weights"""
        print("Starting standard training...")
        start_time = time.perf_counter()
        
        # Create log file
        log_file = os.path.join(self.logs_dir, "training_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"PINN Standard Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epochs: {self.num_epochs}, Use Data Loss: {self.use_data_loss}\n")
            f.write("="*80 + "\n")
        
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 500 == 0:
                loss_info = {
                    'epoch': epoch+1,
                    'total': loss.item()
                }
                self.loss_history.append(loss_info)
                
                progress_msg = f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}'
                print(progress_msg)
                
                with open(log_file, 'a') as f:
                    f.write(progress_msg + "\n")

        total_time = time.perf_counter() - start_time
        
        with open(log_file, 'a') as f:
            f.write("="*80 + "\n")
            f.write(f"Training completed in {total_time:.2f} seconds\n")
        
        print(f"Standard training complete! Total time: {total_time:.2f} seconds")

    def loss_func(self):
        """Standard PINN loss function with optional data loss"""
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
    
        # Combine losses with fixed weights
        loss = loss_f + loss_0 + loss_b + self.data_weight * loss_data
        
        return loss

    def train(self, method='gradient_balanced'):
        """Unified training method"""
        if method == 'standard':
            self.train_standard()
        elif method == 'gradient_balanced':
            self.train_gradient_balanced()
        else:
            print(f"Method '{method}' not implemented. Using gradient_balanced.")
            self.train_gradient_balanced()

    def predict(self, X: torch.Tensor):
        """Make predictions"""
        self.network.eval()
        with torch.no_grad():
            # Ensure input is on same device as model
            model_device = next(self.network.parameters()).device
            X_device = X.to(model_device)
            u_pred = self.network(X_device)
        return u_pred
    
    def compute_l2_error(self):
        """Compute relative L2 error"""
        # Move test data to same device as model
        model_device = next(self.network.parameters()).device
        X_star_device = self.X_star.to(model_device)
        u_star_device = self.u_star.to(model_device)
        
        u_pred = self.predict(X_star_device)
        
        # Convert to numpy for L2 norm calculation
        u_pred_np = u_pred.cpu().numpy()
        u_star_np = u_star_device.cpu().numpy()
        
        error_u = np.linalg.norm(u_star_np - u_pred_np, 2) / np.linalg.norm(u_star_np, 2)
        return error_u

    def save_model(self, name="pinn_model.pth"):
        """Save model to models directory"""
        path = os.path.join(self.models_dir, name)
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")
        
        # Also save full checkpoint with additional info
        checkpoint_path = os.path.join(self.models_dir, "checkpoint.pth")
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_weights': self.loss_weights,
            'loss_history': self.loss_history,
            'weight_history': self.weight_history,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def save_training_info(self, name="training_info.json"):
        """Save training losses, weight history and hyperparameters"""
        path = os.path.join(self.logs_dir, name)

        info = {
            'output_directory': self.output_dir,
            'loss_history': [],
            'weight_history': [],
            'loss_weights': {k: float(v) for k, v in self.loss_weights.items()},
            'num_epochs': int(self.num_epochs),
            'nu': float(self.nu),
            'use_data_loss': bool(self.use_data_loss),
            'data_weight': float(self.data_weight),
            'training_parameters': {
                'N_f': int(self.N_f),
                'N_0': int(self.N_0),
                'N_b': int(self.N_b),
                'data_fraction': 0.1,
                'update_frequency': int(self.update_frequency),
                'alpha': float(self.alpha)
            }
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

    def load_model(self, path):
        """Load model"""
        self.network.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def plot_solution(self, name="prediction.png"):
        """Plot the predicted solution - optimized for HPC"""
        try:
            N_x, N_t = 200, 80
            x = np.linspace(self.x_min, self.x_max, N_x)
            t = np.linspace(self.t_min, self.t_max, N_t)
            X, T = np.meshgrid(x, t)
            XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
            
            XT_tensor = torch.tensor(XT, dtype=torch.float32)
            
            u_pred = self.predict(XT_tensor).cpu().numpy().reshape(N_t, N_x)

            # Create output directory
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # Simple but informative plot
            plt.figure(figsize=(12, 8))
            
            # Contour plot
            ax = plt.subplot(2, 2, (1, 2))
            contour = ax.contourf(T, X, u_pred, levels=50, cmap='coolwarm')
            plt.colorbar(contour, ax=ax, label='u(x,t)')
            ax.set_xlabel('t')
            ax.set_ylabel('x')
            ax.set_title(f'PINN Solution (Epochs: {self.num_epochs}, Data: {self.use_data_loss})')
            
            # Time slices
            time_slices = [0.25, 0.5, 0.75]
            for i, t_slice in enumerate(time_slices):
                ax_slice = plt.subplot(2, 3, 4 + i)
                t_idx = int(t_slice * (N_t - 1))
                u_pred_slice = u_pred[t_idx, :]
                
                ax_slice.plot(x, u_pred_slice, 'b-', label="Predicted", linewidth=2)
                if hasattr(self, 'Exact'):
                    # Match dimensions for comparison
                    if t_idx < self.Exact.shape[0] and len(x) <= self.Exact.shape[1]:
                        exact_slice = self.Exact[t_idx, :len(x)]
                        ax_slice.plot(self.x[:len(x)], exact_slice, 'r--', 
                                    label='Exact', linewidth=2, alpha=0.7)
                ax_slice.set_xlabel('x')
                ax_slice.set_ylabel('u')
                ax_slice.set_title(f't = {t_slice}')
                ax_slice.grid(True, alpha=0.3)
                if i == 0:
                    ax_slice.legend()
            
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Solution plot saved to {save_path}")
            
        except Exception as e:
            print(f"Note: Could not create solution plot: {e}")

    def plot_weight_evolution(self, name="weight_evolution.png"):
        """Plot the evolution of loss weights during training"""
        if not self.weight_history:
            print("No weight history available. Run gradient-balanced training first.")
            return
        
        epochs = [entry['epoch'] for entry in self.weight_history]
        
        plt.figure(figsize=(12, 8))
        
        # Plot weights
        plt.subplot(2, 1, 1)
        if 'r' in self.weight_history[0]['weights']:
            weights_r = [entry['weights']['r'] for entry in self.weight_history]
            plt.plot(epochs, weights_r, 'r-', label='PDE Residual Weight', linewidth=2)
        
        if 'ic' in self.weight_history[0]['weights']:
            weights_ic = [entry['weights']['ic'] for entry in self.weight_history]
            plt.plot(epochs, weights_ic, 'g-', label='Initial Condition Weight', linewidth=2)
        
        if 'bc' in self.weight_history[0]['weights']:
            weights_bc = [entry['weights']['bc'] for entry in self.weight_history]
            plt.plot(epochs, weights_bc, 'b-', label='Boundary Condition Weight', linewidth=2)
        
        if 'data' in self.weight_history[0]['weights']:
            weights_data = [entry['weights']['data'] for entry in self.weight_history]
            plt.plot(epochs, weights_data, 'm-', label='Data Weight', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title(f'Evolution of Loss Weights (Epochs: {self.num_epochs})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot gradient norms if available
        if 'grad_norms' in self.weight_history[0]:
            plt.subplot(2, 1, 2)
            grad_norms_r = [entry['grad_norms'].get('r', 0) for entry in self.weight_history]
            grad_norms_ic = [entry['grad_norms'].get('ic', 0) for entry in self.weight_history]
            grad_norms_bc = [entry['grad_norms'].get('bc', 0) for entry in self.weight_history]
            grad_norms_data = [entry['grad_norms'].get('data', 0) for entry in self.weight_history]
            
            plt.plot(epochs, grad_norms_r, 'r--', label='PDE Residual Grad Norm', linewidth=1.5)
            plt.plot(epochs, grad_norms_ic, 'g--', label='IC Grad Norm', linewidth=1.5)
            plt.plot(epochs, grad_norms_bc, 'b--', label='BC Grad Norm', linewidth=1.5)
            plt.plot(epochs, grad_norms_data, 'm--', label='Data Grad Norm', linewidth=1.5)
            plt.xlabel('Epoch')
            plt.ylabel('Gradient Norm')
            plt.title('Evolution of Gradient Norms')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
    
        os.makedirs(self.plots_dir, exist_ok=True)
        save_path = os.path.join(self.plots_dir, name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"Weight evolution plot saved to {save_path}")
        
    def plot_loss_history(self, name="loss_history.png"):
        """Plot the loss history during training"""
        if not self.loss_history:
            print("No loss history available.")
            return
        
        epochs = [entry['epoch'] for entry in self.loss_history]
        
        plt.figure(figsize=(10, 6))
        
        # Plot total loss
        total_losses = [entry.get('total', 0) for entry in self.loss_history]
        plt.semilogy(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
        
        # Plot component losses if available
        if 'r' in self.loss_history[0]:
            r_losses = [entry.get('r', 0) for entry in self.loss_history]
            plt.semilogy(epochs, r_losses, 'r--', label='PDE Residual Loss', linewidth=1.5)
        
        if 'ic' in self.loss_history[0]:
            ic_losses = [entry.get('ic', 0) for entry in self.loss_history]
            plt.semilogy(epochs, ic_losses, 'g--', label='IC Loss', linewidth=1.5)
        
        if 'bc' in self.loss_history[0]:
            bc_losses = [entry.get('bc', 0) for entry in self.loss_history]
            plt.semilogy(epochs, bc_losses, 'b--', label='BC Loss', linewidth=1.5)
        
        if 'data' in self.loss_history[0] and self.use_data_loss:
            data_losses = [entry.get('data', 0) for entry in self.loss_history]
            plt.semilogy(epochs, data_losses, 'm--', label='Data Loss', linewidth=1.5)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss History (Epochs: {self.num_epochs})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"Loss history plot saved to {save_path}")


def main():
    """Main execution function - everything self-contained"""
    try:
        print("=" * 60)
        print("Starting PINN Training with Data Loss and Weight Balancing")
        print(f"Number of epochs: 5000")
        print("=" * 60)
        
        # Test PyTorch availability
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create and train model - output_dir=None will create timestamped directory
        pinn = PINN(epochs=5000, use_data_loss=True, data_weight=1.0, output_dir=None)
        
        print(f"\nTraining with output in directory: {pinn.output_dir}")
        print("Starting gradient-balanced training...")
        
        pinn.train(method='gradient_balanced')
        
        # Evaluate
        print("\nEvaluating model...")
        try:
            error = pinn.compute_l2_error()
            print(f"Relative L2 Error: {error:.4e}")
            
            # Save error to log file
            error_file = os.path.join(pinn.logs_dir, "error_results.txt")
            with open(error_file, 'w') as f:
                f.write(f"L2 Error: {error:.6e}\n")
                f.write(f"Epochs: {pinn.num_epochs}\n")
                f.write(f"Use Data Loss: {pinn.use_data_loss}\n")
                f.write(f"Data Weight: {pinn.data_weight}\n")
            print(f"Error results saved to: {error_file}")
            
        except Exception as e:
            print(f"Could not compute L2 error: {e}")
        
        # Save results
        print("\nSaving results...")
        pinn.save_model()
        pinn.save_training_info()
        pinn.plot_solution()
        pinn.plot_weight_evolution()
        pinn.plot_loss_history()
        
        # Create a summary file
        summary_file = os.path.join(pinn.output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PINN Training Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Final loss weights: {pinn.loss_weights}\n")
            f.write(f"Number of epochs: {pinn.num_epochs}\n")
            f.write(f"Data loss used: {pinn.use_data_loss}\n")
            f.write(f"Data weight: {pinn.data_weight}\n")
            f.write(f"Output directory: {pinn.output_dir}\n")
            
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"All results saved to: {pinn.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0



if __name__ == "__main__":
    exit(main())

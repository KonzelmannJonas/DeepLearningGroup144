import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # CRITICAL: Non-interactive backend for HPC
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec
import json
    
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
    def __init__(self, loss_weights=None, epochs=2000, use_data_loss=True, data_weight=1.0):
        super(PINN, self).__init__()
        # Set up neural network parameters
        self.network = MLP()
        # REMOVED: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # REMOVED: self.network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.num_epochs = epochs
        
        # Loss weights - initialize with standard values
        self.loss_weights = loss_weights or {'r': 1, 'bc': 1, 'ic': 1}
        if use_data_loss:
            self.loss_weights['data'] = data_weight
            
        self.loss_history = []
        self.weight_history = []
        
        # Set up experiment parameters
        torch.set_default_dtype(torch.float32)
        
        # Spatial and temporal domain boundaries
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

        # Viscosity coefficient
        self.nu = 0.01 / np.pi

        # Collocation points - REDUCED for testing
        self.N_f = 1000  # Reduced from 10000
        self.N_0 = 50    # Reduced from 200
        self.N_b = 50    # Reduced from 200
        
        # Data loss parameters
        self.use_data_loss = use_data_loss
        self.data_weight = data_weight

        # Generate training data
        self.generate_training_data()
        
        # Load ground truth data with fallback
        self.load_ground_truth()
        
        # Setup training data
        self.setup_training_data(data_fraction=0.1)  # Reduced from 0.3 for testing

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

        # Convert to PyTorch tensors - no device specification
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
                '/zhome/de/c/223411/data/burgers_shock.mat'
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
    
        # Combine losses with fixed weights (simplified)
        loss = loss_f + loss_0 + loss_b + self.data_weight * loss_data
        
        return loss

    def train_standard(self):
        """Standard training with fixed weights"""
        print("Starting standard training...")
        start_time = time.perf_counter()
        
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 100 == 0:
                loss_info = {
                    'epoch': epoch+1,
                    'total': loss.item()
                }
                self.loss_history.append(loss_info)
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}')

        total_time = time.perf_counter() - start_time
        print(f"Standard training complete! Total time: {total_time:.2f} seconds")

    def train(self, method='standard'):
        """Unified training method"""
        if method == 'standard':
            self.train_standard()
        else:
            print(f"Method '{method}' not implemented. Using standard training.")
            self.train_standard()

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

    def save_model(self, root="./saved_models", name="pinn_model.pth"):
        """Save model"""
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, name)
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")

    def save_training_info(self, root="./saved_models", name="training_info.json"):
        """Save training losses and hyperparameters to a JSON file."""
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, name)

        info = {
            'loss_history': [],
            'loss_weights': {k: float(v) for k, v in self.loss_weights.items()},
            'num_epochs': int(self.num_epochs),
            'nu': float(self.nu),
            'use_data_loss': bool(self.use_data_loss),
            'data_weight': float(self.data_weight)
        }

        for entry in getattr(self, 'loss_history', []):
            e = entry.copy()
            e['epoch'] = int(e.get('epoch', 0))
            if 'total' in e:
                e['total'] = float(e['total'])
            info['loss_history'].append(e)

        with open(path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Training info saved to {path}")

    def load_model(self, path="./saved_models/pinn_model.pth"):
        """Load model"""
        self.network.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def plot_solution(self, root="./saved_plots/", name="prediction.png"):
        """Plot the predicted solution - simplified for HPC"""
        try:
            N_x, N_t = 100, 50  # Reduced resolution for speed
            x = np.linspace(self.x_min, self.x_max, N_x)
            t = np.linspace(self.t_min, self.t_max, N_t)
            X, T = np.meshgrid(x, t)
            XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
            
            XT_tensor = torch.tensor(XT, dtype=torch.float32)
            
            u_pred = self.predict(XT_tensor).cpu().numpy().reshape(N_t, N_x)

            # Create output directory
            os.makedirs(root, exist_ok=True)
            
            # Simple plot - no subplots for simplicity
            plt.figure(figsize=(10, 8))
            
            # Contour plot
            ax = plt.subplot(2, 1, 1)
            contour = ax.contourf(T, X, u_pred, levels=50, cmap='coolwarm')
            plt.colorbar(contour, ax=ax, label='u(x,t)')
            ax.set_xlabel('t')
            ax.set_ylabel('x')
            ax.set_title('PINN Solution - Burgers Equation with Data Loss')
            
            # Time slice at t=0.5
            ax2 = plt.subplot(2, 1, 2)
            t_idx = int(0.5 * (N_t - 1))
            u_pred_slice = u_pred[t_idx, :]
            
            ax2.plot(x, u_pred_slice, 'b-', label="Predicted", linewidth=2)
            if hasattr(self, 'Exact'):
                exact_slice = self.Exact[t_idx, :len(x)] if t_idx < len(self.Exact) else np.zeros_like(x)
                ax2.plot(self.x[:len(x)], exact_slice, 'r--', label='Exact', linewidth=2, alpha=0.7)
            ax2.set_xlabel('x')
            ax2.set_ylabel('u')
            ax2.set_title('t = 0.5')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            save_path = os.path.join(root, name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {save_path}")
            
        except Exception as e:
            print(f"Note: Could not create plot (non-critical): {e}")


def main():
    """Main execution function - everything self-contained"""
    try:
        # Create output directories
        os.makedirs("./saved_plots", exist_ok=True)
        os.makedirs("./saved_models", exist_ok=True)
        
        print("=" * 60)
        print("Starting PINN Training with Data Loss on HPC")
        print("=" * 60)
        
        # Test PyTorch availability
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create and train model
        print("\nCreating PINN model with data loss...")
        pinn = PINN(epochs=500, use_data_loss=True, data_weight=1.0)  # Reduced epochs for testing
        
        print("\nStarting training...")
        pinn.train(method='standard')
        
        # Evaluate
        print("\nEvaluating model...")
        try:
            error = pinn.compute_l2_error()
            print(f"Relative L2 Error: {error:.4e}")
        except Exception as e:
            print(f"Could not compute L2 error: {e}")
        
        # Save results
        print("\nSaving results...")
        pinn.save_model()
        pinn.save_training_info()
        pinn.plot_solution()
        
        print("\n" + "=" * 60)
        print("SUCCESS: Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

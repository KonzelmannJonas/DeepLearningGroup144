
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

    
class MLP(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 50, 1], activation=nn.Tanh()):
        super().__init__()
        self.activation = activation
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
        self.layers = nn.ModuleList(modules)   

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)

class PINN(nn.Module):
    def __init__(self, loss_weights=None, epochs=5000, lr=0.001, load_from_checkpoint=False, checkpoint_path=None):
        super(PINN, self).__init__()
        
        # Get the script's directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default dtype
        torch.set_default_dtype(torch.float32)
        
        # Domain boundaries
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

        # viscosity coefficient
        self.nu = 0.01 / np.pi

        # Load ground truth for testing (with fallback) - DO THIS FIRST
        self.load_ground_truth()
        
        if load_from_checkpoint and checkpoint_path:
            # Load from checkpoint mode
            self.load_from_checkpoint_mode(checkpoint_path)
        else:
            # Training mode (original code)
            self.training_mode_init(loss_weights, epochs, lr)
    
    def training_mode_init(self, loss_weights=None, epochs=5000, lr=0.001):
        """Initialize for training (original behavior)"""
        # set up neural network parameters
        self.network = MLP()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.num_epochs = epochs
        
        # Loss weights
        self.loss_weights = loss_weights or {'r': 1.0, 'bc': 1.0, 'ic': 1.0}
        self.loss_history = []
        self.weight_history = []
        
        # Gradient balancing parameters
        self.update_frequency = 100
        self.alpha = 0.9

        # collocation points - REDUCED for testing
        self.N_f = 1000  # Reduced from 10000
        self.N_0 = 50    # Reduced from 200
        self.N_b = 50    # Reduced from 200

        # Generate training data
        self.generate_training_data()
        
        print("Initialized in TRAINING mode")
    
    def load_from_checkpoint_mode(self, checkpoint_path):
        """Initialize by loading from checkpoint"""
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Load the checkpoint
        if not os.path.exists(checkpoint_path):
            # Try to find it in the script directory
            checkpoint_path = os.path.join(self.script_dir, checkpoint_path)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        
        # Load network
        self.network = MLP()
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()  # Set to evaluation mode
        
        # Load other parameters
        self.loss_weights = checkpoint.get('loss_weights', {'r': 1.0, 'bc': 1.0, 'ic': 1.0})
        self.weight_history = checkpoint.get('weight_history', [])
        self.loss_history = checkpoint.get('loss_history', [])
        
        # Load training params
        training_params = checkpoint.get('training_params', {})
        self.num_epochs = training_params.get('num_epochs', 5000)
        self.update_frequency = training_params.get('update_frequency', 100)
        self.alpha = training_params.get('alpha', 0.9)
        
        # Create a dummy optimizer (not used for evaluation)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        
        # Don't generate training data for evaluation mode
        self.N_f = 0
        self.N_0 = 0
        self.N_b = 0
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Loss weights: {self.loss_weights}")
        print(f"✓ Training ran for {self.num_epochs} epochs")
        if self.loss_history:
            print(f"✓ Loss history has {len(self.loss_history)} entries")
            final_loss = self.loss_history[-1]
            print(f"✓ Final total loss: {final_loss['total']:.6e}")

    def generate_training_data(self):
        """Generate collocation points and initial/boundary conditions"""
        # Collocation points for PDE residual
        X_f = np.random.rand(self.N_f, 2)
        X_f[:, 0] = X_f[:, 0] * (self.x_max - self.x_min) + self.x_min
        X_f[:, 1] = X_f[:, 1] * (self.t_max - self.t_min) + self.t_min
        self.X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True)

        # Initial condition
        x0 = np.linspace(self.x_min, self.x_max, self.N_0)[:, None]
        t0 = np.zeros_like(x0)
        u0 = -np.sin(np.pi * x0)
        self.x0 = torch.tensor(x0, dtype=torch.float32)
        self.t0 = torch.tensor(t0, dtype=torch.float32)
        self.u0 = torch.tensor(u0, dtype=torch.float32)

        # Boundary conditions
        tb = np.linspace(self.t_min, self.t_max, self.N_b)[:, None]
        xb_left = np.ones_like(tb) * self.x_min
        xb_right = np.ones_like(tb) * self.x_max
        self.tb = torch.tensor(tb, dtype=torch.float32)
        self.xb_left = torch.tensor(xb_left, dtype=torch.float32)
        self.xb_right = torch.tensor(xb_right, dtype=torch.float32)

    def load_ground_truth(self):
        """Load ground truth data for testing with fallback"""
        try:
            # Try multiple possible paths for the data file
            possible_paths = [
                'burgers_shock.mat',
                './data/burgers_shock.mat',
                '../data/burgers_shock.mat',
                '/zhome/de/c/223411/data/burgers_shock.mat'  # Your home directory
            ]
            
            data_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            
            if data_path:
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
                print(f"Loaded ground truth from {data_path}")
            else:
                print("Warning: Ground truth data file not found. Creating synthetic data for testing.")
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

    def compute_component_gradients(self, loss_dict):
        """Compute gradient norms for each loss component"""
        grad_norms = {}
        
        for name, loss in loss_dict.items():
            # Clear previous gradients
            self.optimizer.zero_grad()
            
            # Compute gradients for this specific loss component
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=list(self.network.parameters()),
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

    def update_loss_weights_standard(self, grad_norms):
        """Standard gradient norm balancing"""
        new_weights = self.loss_weights.copy()
        eps = 1e-8
        
        ref_norm = max(grad_norms.get('r', eps), grad_norms.get('bc', eps), grad_norms.get('ic', eps))
        
        target_ratio_r = ref_norm / (grad_norms['r'] + eps)
        target_ratio_ic = ref_norm / (grad_norms['ic'] + eps)
            
        new_weights['r'] = (self.alpha * self.loss_weights['r'] + 
                          (1 - self.alpha) * target_ratio_r)
        new_weights['ic'] = (self.alpha * self.loss_weights['ic'] + 
                           (1 - self.alpha) * target_ratio_ic)
        
        new_weights['bc'] = 1.0
        
        # Clamp weights
        max_weight = 100.0
        min_weight = 0.01
        for name in new_weights:
            new_weights[name] = np.clip(new_weights[name], min_weight, max_weight)
        
        self.loss_weights = new_weights

    def compute_loss_components(self):
        """Compute individual loss components separately"""
        # PDE residual loss
        f_pred = self.pde_residual(self.X_f)
        loss_r = torch.mean(f_pred**2)

        # Initial condition loss
        u0_pred = self.network(torch.cat([self.x0, self.t0], dim=1))
        loss_ic = torch.mean((u0_pred - self.u0)**2)

        # Boundary condition loss
        u_left_pred = self.network(torch.cat([self.xb_left, self.tb], dim=1))
        u_right_pred = self.network(torch.cat([self.xb_right, self.tb], dim=1))
        loss_bc = torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2)
    
        return {'r': loss_r, 'ic': loss_ic, 'bc': loss_bc}

    def pde_residual(self, X):
        """Compute Burgers' equation residual"""
        x = X[:, 0:1] 
        t = X[:, 1:2] 
        u = self.network(torch.cat([x, t], dim=1))

        # Compute gradients
        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                           create_graph=True, retain_graph=True)[0]
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                           create_graph=True, retain_graph=True)[0]
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                            create_graph=True, retain_graph=True)[0]

        # Burgers' equation residual
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def train_gradient_balanced(self):
        print(f"Training on device: {next(self.network.parameters()).device}")
        """Training with gradient norm balancing"""
        print("Starting gradient-balanced training...")
        print(f"Initial weights: {self.loss_weights}")
        start_time = time.perf_counter()
    
        for epoch in range(self.num_epochs):
            # Compute individual loss components
            loss_dict = self.compute_loss_components()
        
            # Update weights periodically based on gradient norms
            if epoch % self.update_frequency == 0:
                with torch.no_grad():
                    grad_norms = self.compute_component_gradients(loss_dict)
                    self.update_loss_weights_standard(grad_norms)
                    
                    # Record weight history
                    self.weight_history.append({
                        'epoch': epoch,
                        'weights': self.loss_weights.copy(),
                        'grad_norms': grad_norms,
                        'losses': {k: v.item() for k, v in loss_dict.items()}
                    })
                    
                    # Debug output every 500 epochs
                    if epoch % 500 == 0:
                        print(f"\nEpoch {epoch}:")
                        print(f"  Losses: { {k: f'{v.item():.2e}' for k, v in loss_dict.items()} }")
                        print(f"  Grad Norms: { {k: f'{v:.2e}' for k, v in grad_norms.items()} }")
                        print(f"  Weights: {self.loss_weights}")
        
            # Compute weighted total loss
            total_loss = (self.loss_weights['r'] * loss_dict['r'] + 
                         self.loss_weights['ic'] * loss_dict['ic'] + 
                         self.loss_weights['bc'] * loss_dict['bc'])
        
            # Standard optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging
            if (epoch+1) % 100 == 0:
                loss_info = {
                    'epoch': epoch+1,
                    'total': total_loss.item(),
                    'r': loss_dict['r'].item(),
                    'ic': loss_dict['ic'].item(),
                    'bc': loss_dict['bc'].item(),
                    'weights': self.loss_weights.copy()
                }
                self.loss_history.append(loss_info)
                print(f'Epoch {epoch+1}/{self.num_epochs}, '
                      f'Total Loss: {total_loss.item():.5e}')

        total_time = time.perf_counter() - start_time
        print(f"Gradient-balanced training complete! Total time: {total_time:.2f} seconds")
        print(f"Final weights: {self.loss_weights}")

    def compute_l2_error(self):
        """Compute relative L2 error using loaded ground truth data"""
        # Make sure we have ground truth data
        if not hasattr(self, 'X_star') or not hasattr(self, 'u_star'):
            print("Warning: No ground truth data loaded. Cannot compute L2 error.")
            return float('nan')
        
        # Set model to evaluation mode
        self.network.eval()
        
        # Move test data to same device as model
        model_device = next(self.network.parameters()).device
        X_star_device = self.X_star.to(model_device)
        u_star_device = self.u_star.to(model_device)
        
        # Make predictions
        with torch.no_grad():
            u_pred = self.network(X_star_device)
        
        # Compute L2 error
        error_u = torch.norm(u_star_device - u_pred, 2) / torch.norm(u_star_device, 2)
        
        return error_u.item()

    def predict(self, X):
        """Make predictions"""
        self.network.eval()
        with torch.no_grad():
            u_pred = self.network(X)
        return u_pred

    def plot_solution(self, name="prediction.png"):
        """Plot the predicted solution - simplified for HPC"""
        try:
            N_x, N_t = 100, 50  # Reduced for speed
            x = np.linspace(self.x_min, self.x_max, N_x)
            t = np.linspace(self.t_min, self.t_max, N_t)
            X, T = np.meshgrid(x, t)
            XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
            
            XT_tensor = torch.tensor(XT, dtype=torch.float32)
            model_device = next(self.network.parameters()).device
            XT_tensor = XT_tensor.to(model_device)
            
            u_pred = self.predict(XT_tensor).cpu().numpy().reshape(N_t, N_x)

            # Simple plot
            plt.figure(figsize=(10, 6))
            plt.contourf(T, X, u_pred, levels=50, cmap='coolwarm')
            plt.colorbar(label='u(x,t)')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title('PINN Solution - Burgers Equation')
            plt.tight_layout()
            
            save_path = os.path.join(self.script_dir, name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {save_path}")
            
        except Exception as e:
            print(f"Note: Could not create plot (non-critical): {e}")

    def save_training_history_txt(self, name="training_history.txt"):
        """Save training history as a text file"""
        path = os.path.join(self.script_dir, name)
    
        try:
            # Use UTF-8 encoding to be safe, but avoid Unicode characters
            with open(path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("=" * 80 + "\n")
                f.write("PINN TRAINING HISTORY\n")
                f.write("=" * 80 + "\n\n")
            
                # Write configuration - use ASCII characters only
                f.write("CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total epochs: {self.num_epochs}\n")
                f.write(f"Learning rate: 0.001\n")
                f.write(f"Viscosity (nu): {self.nu}\n")
                f.write(f"Domain: x in [{self.x_min}, {self.x_max}], t in [{self.t_min}, {self.t_max}]\n")  # ASCII version
                f.write(f"Collocation points: {self.N_f}\n")
                f.write(f"Initial condition points: {self.N_0}\n")
                f.write(f"Boundary condition points: {self.N_b}\n")
                f.write(f"Final loss weights: {self.loss_weights}\n")
                f.write("\n")
            
                # Write loss history (every 100 epochs)
                if self.loss_history:
                    f.write("LOSS HISTORY (every 100 epochs):\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{'Epoch':>6} {'Total Loss':>15} {'PDE Loss':>15} {'IC Loss':>15} {'BC Loss':>15} {'Weights (r,ic,bc)'}\n")
                    f.write("-" * 80 + "\n")
                
                    for entry in self.loss_history:
                        epoch = entry['epoch']
                        total = entry['total']
                        r_loss = entry['r']
                        ic_loss = entry['ic']
                        bc_loss = entry['bc']
                        weights = entry['weights']
                        weights_str = f"({weights['r']:.2f}, {weights['ic']:.2f}, {weights['bc']:.2f})"
                    
                        f.write(f"{epoch:6d} {total:15.6e} {r_loss:15.6e} {ic_loss:15.6e} {bc_loss:15.6e} {weights_str}\n")
                    f.write("\n")
            
                # Write weight adjustment history (every update_frequency epochs)
                if self.weight_history:
                    f.write(f"WEIGHT ADJUSTMENT HISTORY (every {self.update_frequency} epochs):\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{'Epoch':>6} {'Weights (r,ic,bc)':>25} {'Grad Norms (r,ic,bc)':>30} {'Losses (r,ic,bc)':>30}\n")
                    f.write("-" * 85 + "\n")
                
                    for entry in self.weight_history:
                        epoch = entry['epoch']
                        weights = entry['weights']
                        grad_norms = entry['grad_norms']
                        losses = entry['losses']
                    
                        weights_str = f"({weights['r']:.2f}, {weights['ic']:.2f}, {weights['bc']:.2f})"
                        grad_str = f"({grad_norms['r']:.2e}, {grad_norms['ic']:.2e}, {grad_norms['bc']:.2e})"
                        loss_str = f"({losses['r']:.2e}, {losses['ic']:.2e}, {losses['bc']:.2e})"
                    
                        f.write(f"{epoch:6d} {weights_str:>25} {grad_str:>30} {loss_str:>30}\n")
                    f.write("\n")
            
                # Write final statistics
                f.write("FINAL STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Training completed successfully\n")
                f.write(f"Final loss weights: {self.loss_weights}\n")
                if self.loss_history:
                    final_loss = self.loss_history[-1]
                    f.write(f"Final losses - Total: {final_loss['total']:.6e}, "
                       f"PDE: {final_loss['r']:.6e}, "
                       f"IC: {final_loss['ic']:.6e}, "
                       f"BC: {final_loss['bc']:.6e}\n")
            
                # Compute and write L2 error
                l2_error = self.compute_l2_error()
                f.write(f"Relative L2 Error: {l2_error:.6e}\n")
            
            print(f"Training history saved to {path}")
        
        except Exception as e:
            print(f"Error saving training history: {e}")

    def save_model(self, name="pinn_model.pth"):
        """Save model parameters AND training history"""
        path = os.path.join(self.script_dir, name)
        
        # Save everything including training history
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_weights': self.loss_weights,
            'weight_history': self.weight_history,
            'loss_history': self.loss_history,
            'training_params': {
                'num_epochs': self.num_epochs,
                'update_frequency': self.update_frequency,
                'alpha': self.alpha,
                'nu': self.nu,
                'x_min': self.x_min,
                'x_max': self.x_max,
                't_min': self.t_min,
                't_max': self.t_max
            }
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path="pinn_model.pth"):
        """Load model parameters"""
        # Get full path
        if not os.path.isabs(path):
            path = os.path.join(self.script_dir, path)
            
        save_dict = torch.load(path, weights_only=False)
    
        self.network.load_state_dict(save_dict['network_state_dict'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        self.loss_weights = save_dict.get('loss_weights', {'r': 1.0, 'bc': 1.0, 'ic': 1.0})
        self.weight_history = save_dict.get('weight_history', [])
        self.loss_history = save_dict.get('loss_history', [])
        print(f"Model loaded from {path}")

    def plot_weight_evolution(self, name="weight_evolution.png"):
        """Plot the evolution of loss weights during training"""
        if not self.weight_history:
            print("No weight history available.")
            return
        
        epochs = [entry['epoch'] for entry in self.weight_history]
        weights_r = [entry['weights']['r'] for entry in self.weight_history]
        weights_ic = [entry['weights']['ic'] for entry in self.weight_history]
        weights_bc = [entry['weights']['bc'] for entry in self.weight_history]
    
        plt.figure(figsize=(12, 8))
        
        # Plot weights
        plt.subplot(2, 1, 1)
        plt.plot(epochs, weights_r, 'r-', label='PDE Residual Weight', linewidth=2)
        plt.plot(epochs, weights_ic, 'g-', label='Initial Condition Weight', linewidth=2)
        plt.plot(epochs, weights_bc, 'b-', label='Boundary Condition Weight', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title('Evolution of Loss Weights (Gradient Norm Balancing)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot gradient norms
        plt.subplot(2, 1, 2)
        grad_norms_r = [entry['grad_norms']['r'] for entry in self.weight_history]
        grad_norms_ic = [entry['grad_norms']['ic'] for entry in self.weight_history]
        grad_norms_bc = [entry['grad_norms']['bc'] for entry in self.weight_history]
        
        plt.plot(epochs, grad_norms_r, 'r--', label='PDE Residual Grad Norm', linewidth=2)
        plt.plot(epochs, grad_norms_ic, 'g--', label='IC Grad Norm', linewidth=2)
        plt.plot(epochs, grad_norms_bc, 'b--', label='BC Grad Norm', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Evolution of Gradient Norms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"Weight evolution plot saved to {save_path}")
        
    def quick_data_check(self):
        """Quick check of data shapes and ranges"""
        print("\nQuick Data Check:")
        if hasattr(self, 'X_f'):
            print(f"PDE collocation points: {self.X_f.shape}")
        if hasattr(self, 'x0'):
            print(f"Initial condition: x={self.x0.shape}, u={self.u0.shape}")
        if hasattr(self, 'tb'):
            print(f"Boundary points: {self.tb.shape}")
        print(f"Ground truth: {getattr(self, 'X_star', 'Not loaded').shape if hasattr(self, 'X_star') else 'Not loaded'}")


def main():
    """Main execution function - can either train or load from checkpoint"""
    try:
        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("=" * 60)
        print("PINN FOR BURGERS EQUATION")
        print("=" * 60)
        print(f"Script location: {script_dir}")
        
        # Test PyTorch availability
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # ====================================================================
        # CHOOSE MODE: TRAIN NEW MODEL OR LOAD EXISTING MODEL
        # ====================================================================
        MODE = "LOAD"  # Change to "TRAIN" to train a new model
        
        if MODE == "TRAIN":
            print("\n" + "="*50)
            print("TRAINING NEW MODEL")
            print("="*50)
            
            # Create and train model
            print("\nCreating PINN model...")
            pinn = PINN(epochs=5000, lr=0.001) 
            
            print("\nStarting training...")
            pinn.train_gradient_balanced()
            
            # Evaluate
            print("\nEvaluating model...")
            error = pinn.compute_l2_error()
            print(f"Relative L2 Error: {error:.4e}")
            
            # Save results
            print("\nSaving results...")
            pinn.save_model()
            pinn.save_training_history_txt()
            pinn.plot_solution()
            pinn.plot_weight_evolution()
            
        elif MODE == "LOAD":
            print("\n" + "="*50)
            print("LOADING EXISTING MODEL")
            print("="*50)
            
            # Check if model file exists
            model_path = "pinn_model.pth"
            full_path = os.path.join(script_dir, model_path)
            
            if not os.path.exists(full_path):
                print(f"Model file not found: {full_path}")
                print("Looking for model files in directory:")
                files = os.listdir(script_dir)
                pth_files = [f for f in files if f.endswith('.pth')]
                if pth_files:
                    print(f"Found .pth files: {pth_files}")
                    model_path = pth_files[0]
                    print(f"Using: {model_path}")
                else:
                    print("No .pth files found. Switching to training mode.")
                    MODE = "TRAIN"
            
            if MODE == "LOAD":
                # Load existing model
                print(f"\nLoading model from: {model_path}")
                pinn = PINN(load_from_checkpoint=True, checkpoint_path=model_path)
                
                # Evaluate using loaded ground truth data
                print("\nEvaluating loaded model...")
                error = pinn.compute_l2_error()
                print(f"Relative L2 Error: {error:.4e}")
                
                # Make some predictions
                print("\n" + "="*50)
                print("TEST PREDICTIONS")
                print("="*50)
                
                test_points = [
                    (0.0, 0.0),
                    (0.5, 0.5), 
                    (-0.5, 0.25),
                    (0.0, 1.0)
                ]
                
                for x, t in test_points:
                    X_test = torch.tensor([[x, t]], dtype=torch.float32)
                    with torch.no_grad():
                        u_pred = pinn.network(X_test)
                    print(f"u({x:.2f}, {t:.2f}) = {u_pred.item():.6f}")
                
                # Create plots
                print("\nCreating plots...")
                pinn.plot_solution("loaded_prediction.png")
                if pinn.weight_history:
                    pinn.plot_weight_evolution("loaded_weight_evolution.png")
                
                # Print training summary
                if pinn.loss_history:
                    print("\n" + "="*50)
                    print("TRAINING SUMMARY FROM LOADED MODEL")
                    print("="*50)
                    final_loss = pinn.loss_history[-1]
                    print(f"Final epoch: {final_loss['epoch']}")
                    print(f"Final total loss: {final_loss['total']:.6e}")
                    print(f"Final PDE loss: {final_loss['r']:.6e}")
                    print(f"Final IC loss: {final_loss['ic']:.6e}")
                    print(f"Final BC loss: {final_loss['bc']:.6e}")
                    print(f"Final weights: {final_loss['weights']}")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Operation completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
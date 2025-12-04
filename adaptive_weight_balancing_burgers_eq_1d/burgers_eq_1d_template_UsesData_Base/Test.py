import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
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
from pinn import PINN


class PINNTester:
    def __init__(self, pinn_model):
        self.model = pinn_model
        self.results = {}
        
    def add_noise_to_data(self, data, noise_level=0.05, noise_type='gaussian'):
        """
        Add different types of noise to test data
        
        Args:
            data: Clean data tensor
            noise_level: Standard deviation for gaussian, or percentage for uniform
            noise_type: 'gaussian', 'uniform', 'outlier', 'mixed'
        """
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * np.std(data_np), data_np.shape)
            noisy_data = data_np + noise
            
        elif noise_type == 'uniform':
            data_range = np.ptp(data_np)  # Peak-to-peak range
            noise = np.random.uniform(-noise_level * data_range, 
                                    noise_level * data_range, 
                                    data_np.shape)
            noisy_data = data_np + noise
            
        elif noise_type == 'outlier':
            noisy_data = data_np.copy()
            # Randomly select points to be outliers
            outlier_mask = np.random.random(data_np.shape) < noise_level
            outlier_strength = 5 * np.std(data_np)  # 5 sigma outliers
            noisy_data[outlier_mask] += outlier_strength * np.random.randn(np.sum(outlier_mask))
            
        elif noise_type == 'mixed':
            # Combination of gaussian noise and outliers
            gaussian_noise = np.random.normal(0, 0.5 * noise_level * np.std(data_np), data_np.shape)
            noisy_data = data_np + gaussian_noise
            
            # Add outliers
            outlier_mask = np.random.random(data_np.shape) < 0.1 * noise_level
            outlier_strength = 5 * np.std(data_np)
            noisy_data[outlier_mask] += outlier_strength * np.random.randn(np.sum(outlier_mask))
        
        return torch.tensor(noisy_data, dtype=torch.float32).to(self.model.device)
    
    def test_noise_robustness(self, noise_levels=[0.01, 0.05, 0.1, 0.2], 
                            noise_types=['gaussian', 'uniform', 'outlier']):
        """
        Test model performance across different noise levels and types
        """
        print("=== Testing Noise Robustness ===")
        
        results = {}
        clean_error = self.model.compute_l2_error()
        print(f"Clean data L2 error: {clean_error:.6f}")
        
        for noise_type in noise_types:
            results[noise_type] = {}
            print(f"\n--- Noise Type: {noise_type} ---")
            
            for level in noise_levels:
                # Create noisy test data
                X_noisy = self.add_noise_to_data(self.model.X_star, level, noise_type)
                u_noisy = self.add_noise_to_data(self.model.u_star, level, noise_type)
                
                # Predict on noisy data
                u_pred = self.model.predict(X_noisy)
                
                # Compute errors
                mse = torch.mean((u_pred - u_noisy)**2).item()
                l2_error = torch.norm(u_pred - u_noisy, 2).item() / torch.norm(u_noisy, 2).item()
                
                # Also compute error relative to clean data (true performance)
                true_mse = torch.mean((u_pred - self.model.u_star)**2).item()
                true_l2 = torch.norm(u_pred - self.model.u_star, 2).item() / torch.norm(self.model.u_star, 2).item()
                
                results[noise_type][level] = {
                    'mse_noisy_target': mse,
                    'l2_noisy_target': l2_error,
                    'mse_clean_target': true_mse,
                    'l2_clean_target': true_l2
                }
                
                print(f"Noise level {level:.3f}: "
                      f"L2 (noisy target) = {l2_error:.6f}, "
                      f"L2 (clean target) = {true_l2:.6f}")
        
        self.results['noise_robustness'] = results
        return results
    
    def test_data_fraction_robustness(self, data_fractions=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """
        Test how model performs when trained with different amounts of data
        (Useful if your model was trained with data loss)
        """
        print("\n=== Testing Data Fraction Robustness ===")
        
        if not hasattr(self.model, 'X_data'):
            print("Model was not trained with data loss - skipping data fraction test")
            return None
        
        results = {}
        
        for fraction in data_fractions:
            # Create a new model instance
            temp_model = PINN(epochs=500, use_data_loss=True, data_weight=1.0)
            temp_model.setup_training_data(data_fraction=fraction)
            
            # Quick training (fewer epochs for testing)
            temp_model.train(method='standard')
            
            # Test performance
            error = temp_model.compute_l2_error()
            results[fraction] = error
            
            print(f"Data fraction {fraction}: L2 error = {error:.6f}")
        
        self.results['data_fraction'] = results
        return results
    
    def compute_generalization_gap(self, train_points_ratio=0.1):
        """
        Compute generalization gap by comparing performance on training vs test points
        """
        print("\n=== Computing Generalization Gap ===")
        
        # Sample training points (if model was trained with data)
        if hasattr(self.model, 'X_data'):
            X_train = self.model.X_data
            u_train = self.model.u_data
        else:
            # Use collocation points as proxy for training data
            n_train = int(self.model.X_f.shape[0] * train_points_ratio)
            indices = np.random.choice(self.model.X_f.shape[0], n_train, replace=False)
            X_train = self.model.X_f[indices]
            u_train = self.model.network(X_train).detach()
        
        # Training error
        u_pred_train = self.model.predict(X_train)
        train_error = torch.mean((u_pred_train - u_train)**2).item()
        
        # Test error (on clean full dataset)
        test_error = torch.mean((self.model.predict(self.model.X_star) - self.model.u_star)**2).item()
        
        generalization_gap = test_error - train_error
        generalization_ratio = test_error / train_error if train_error > 0 else float('inf')
        
        results = {
            'train_error': train_error,
            'test_error': test_error,
            'generalization_gap': generalization_gap,
            'generalization_ratio': generalization_ratio
        }
        
        print(f"Train MSE: {train_error:.2e}")
        print(f"Test MSE: {test_error:.2e}")
        print(f"Generalization gap: {generalization_gap:.2e}")
        print(f"Generalization ratio: {generalization_ratio:.2f}")
        
        if generalization_ratio > 2.0:
            print("🚨 WARNING: High generalization ratio - possible overfitting")
        elif generalization_ratio < 1.0:
            print("✅ Good: Test error lower than training error")
        else:
            print("✅ Reasonable generalization")
        
        self.results['generalization'] = results
        return results
    
    def sensitivity_analysis(self, n_samples=1000):
        """
        Perform sensitivity analysis by adding small perturbations to inputs
        """
        print("\n=== Sensitivity Analysis ===")
        
        # Sample random points from domain
        n_test = min(n_samples, self.model.X_star.shape[0])
        indices = np.random.choice(self.model.X_star.shape[0], n_test, replace=False)
        X_test = self.model.X_star[indices]
        
        # Add small perturbations
        perturbations = [0.001, 0.005, 0.01, 0.02]
        sensitivity_results = {}
        
        for perturb in perturbations:
            output_changes = []
            
            for i in range(n_test):
                # Original prediction
                orig_pred = self.model.predict(X_test[i:i+1])
                
                # Perturbed prediction
                X_perturbed = X_test[i:i+1].clone()
                X_perturbed += torch.randn_like(X_perturbed) * perturb
                pert_pred = self.model.predict(X_perturbed)
                
                # Relative change
                rel_change = torch.abs(pert_pred - orig_pred) / (torch.abs(orig_pred) + 1e-8)
                output_changes.append(rel_change.item())
            
            sensitivity_results[perturb] = {
                'mean_sensitivity': np.mean(output_changes),
                'std_sensitivity': np.std(output_changes),
                'max_sensitivity': np.max(output_changes)
            }
            
            print(f"Perturbation {perturb:.4f}: "
                  f"Mean sensitivity = {np.mean(output_changes):.6f}, "
                  f"Max = {np.max(output_changes):.6f}")
        
        self.results['sensitivity'] = sensitivity_results
        return sensitivity_results
    
    def plot_noise_robustness_results(self, save_path="./saved_plots/", show=False):
        """Plot noise robustness test results"""
        if 'noise_robustness' not in self.results:
            print("No noise robustness results available. Run test_noise_robustness() first.")
            return
        
        results = self.results['noise_robustness']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        noise_levels = list(next(iter(results.values())).keys())
        
        # Plot 1: L2 error with noisy targets
        for i, (noise_type, data) in enumerate(results.items()):
            errors = [data[level]['l2_noisy_target'] for level in noise_levels]
            axes[0].plot(noise_levels, errors, 'o-', label=noise_type, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('L2 Error (Noisy Target)')
        axes[0].set_title('Performance on Noisy Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Plot 2: L2 error with clean targets (true performance)
        for i, (noise_type, data) in enumerate(results.items()):
            errors = [data[level]['l2_clean_target'] for level in noise_levels]
            axes[1].plot(noise_levels, errors, 'o-', label=noise_type, linewidth=2, markersize=8)
        
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('L2 Error (Clean Target)')
        axes[1].set_title('True Performance Despite Noise')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Plot 3: Performance degradation
        clean_error = self.model.compute_l2_error()
        degradation_data = {}
        
        for noise_type, data in results.items():
            degradations = [data[level]['l2_clean_target'] / clean_error for level in noise_levels]
            degradation_data[noise_type] = degradations
        
        for noise_type, degradations in degradation_data.items():
            axes[2].plot(noise_levels, degradations, 'o-', label=noise_type, linewidth=2, markersize=8)
        
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (no noise)')
        axes[2].set_xlabel('Noise Level')
        axes[2].set_ylabel('Error Ratio (Noisy/Clean)')
        axes[2].set_title('Performance Degradation')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Bar chart of worst-case performance
        worst_case_errors = []
        noise_types = list(results.keys())
        
        for noise_type in noise_types:
            max_error = max([results[noise_type][level]['l2_clean_target'] for level in noise_levels])
            worst_case_errors.append(max_error)
        
        axes[3].bar(noise_types, worst_case_errors, color=['red', 'blue', 'green', 'orange'])
        axes[3].set_ylabel('Worst-case L2 Error')
        axes[3].set_title('Worst-case Performance by Noise Type')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'noise_robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close('all')
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive overfitting and robustness report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL ROBUSTNESS REPORT")
        print("="*60)
        
        # Run all tests if not already run
        if 'noise_robustness' not in self.results:
            self.test_noise_robustness()
        
        if 'generalization' not in self.results:
            self.compute_generalization_gap()
        
        if 'sensitivity' not in self.results:
            self.sensitivity_analysis()
        
        # Summary metrics
        clean_error = self.model.compute_l2_error()
        gen_ratio = self.results['generalization']['generalization_ratio']
        
        # Noise robustness score (average performance degradation at 10% noise)
        noise_results = self.results['noise_robustness']
        robustness_scores = []
        for noise_type, data in noise_results.items():
            if 0.1 in data:
                degradation = data[0.1]['l2_clean_target'] / clean_error
                robustness_scores.append(degradation)
        
        avg_robustness = np.mean(robustness_scores) if robustness_scores else float('inf')
        
        # Sensitivity score
        sensitivity = self.results['sensitivity']
        mean_sensitivities = [data['mean_sensitivity'] for data in sensitivity.values()]
        avg_sensitivity = np.mean(mean_sensitivities)
        
        print(f"\nSUMMARY METRICS:")
        print(f"✅ Clean L2 Error: {clean_error:.6f}")
        print(f"📊 Generalization Ratio: {gen_ratio:.2f}")
        print(f"🛡️  Noise Robustness Score: {avg_robustness:.2f} (lower is better)")
        print(f"🎯 Average Sensitivity: {avg_sensitivity:.6f}")
        
        print(f"\nOVERFITTING ASSESSMENT:")
        if gen_ratio > 2.0:
            print("🚨 HIGH RISK of overfitting")
        elif gen_ratio > 1.5:
            print("⚠️  MODERATE RISK of overfitting")
        else:
            print("✅ LOW RISK of overfitting")
        
        print(f"\nROBUSTNESS ASSESSMENT:")
        if avg_robustness > 3.0:
            print("🚨 POOR noise robustness")
        elif avg_robustness > 2.0:
            print("⚠️  MODERATE noise robustness")
        else:
            print("✅ GOOD noise robustness")
        
        return {
            'clean_error': clean_error,
            'generalization_ratio': gen_ratio,
            'noise_robustness_score': avg_robustness,
            'sensitivity_score': avg_sensitivity
        }
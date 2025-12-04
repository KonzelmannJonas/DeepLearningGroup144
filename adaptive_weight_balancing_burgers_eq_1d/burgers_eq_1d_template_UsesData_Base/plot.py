from pinn import PINN
from Test import PINNTester

def main():
    pinn = PINN()
    pinn.load_model()
    pinn.plot_weight_evolution()
    pinn.plot_solution()
    

    error = pinn.compute_l2_error()
    print(f"L2 Error: {error:.6f}")

    # After training your model


# Load your trained model
    pinn = PINN()
    pinn.load_model()  # Load your trained model

# Initialize tester
    tester = PINNTester(pinn)

# Run comprehensive tests
    print("Running comprehensive robustness tests...")

# 1. Test noise robustness
    tester.test_noise_robustness(noise_levels=[0.01, 0.05, 0.1, 0.2, 0.3])

# 2. Test generalization
    tester.compute_generalization_gap()

# 3. Sensitivity analysis
    tester.sensitivity_analysis()

# 4. Plot results
    tester.plot_noise_robustness_results()

# 5. Generate comprehensive report
    report = tester.generate_comprehensive_report()

# Save results
    import pickle
    with open('./saved_models/robustness_report.pkl', 'wb') as f:
        pickle.dump(tester.results, f)

    print("\nTesting complete! Check the generated plots and report.")
    
if __name__ == "__main__":
    main()
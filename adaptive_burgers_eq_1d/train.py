from pinn import PINN_Burgers

def main():
    # adaptive sampling
    pinn = PINN_Burgers(is_adaptive=True)    
    pinn.train()
    pinn.save_model()
    print(f"N_f: {pinn.N_f}")
    print(f"N_f from tensor: {pinn.X_f.shape[0]}")
    error_l2 = pinn.compute_l2_error()
    print(f"relative L2 error: {error_l2:.6e}")
    
if __name__ == "__main__":
    main()

## Warning: Ground truth data './data/burgers_shock.mat' not found. L2 error computation will be skipped.
# Guys I tried the training without real data XD
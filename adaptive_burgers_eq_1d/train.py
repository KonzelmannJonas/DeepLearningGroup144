from pinn import PINN

def main():
    pinn = PINN()
    # pinn.is_adaptive = False
    pinn.train()
    pinn.save_model()
    print(f"N_f: {pinn.N_f}")
    print(f"N_f from tensor: {pinn.X_f.shape[0]}")
    error_l2 = pinn.compute_l2_error()
    print(f"relative L2 error: {error_l2:.6e}")
    
if __name__ == "__main__":
    main()
from pinn import PINN

def main():
    # adaptive sampling
    pinn = PINN(is_adaptive=True)    
    pinn.train()
    
    # print(f"N_f: {pinn.N_f}")
    # print(f"N_f from tensor: {pinn.X_f.shape[0]}")
    error_l2 = pinn.compute_l2_error()
    print(f"relative L2 error: {error_l2:.6e}")
    pinn.plot_solution()
    pinn.save_model()
if __name__ == "__main__":
    main()

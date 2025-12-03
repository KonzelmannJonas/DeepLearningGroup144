from pinn import PINN

def main():
    pinn = PINN()
    pinn.train()
    error_l2 = pinn.compute_l2_error()
    print(f"relative L2 error: {error_l2:.6e}")
    pinn.save_plot_parameters()
    
if __name__ == "__main__":
    main()

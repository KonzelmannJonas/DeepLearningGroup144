from pinn import PINN

def main():
    pinn = PINN()
    pinn.load_model(path="./saved_models/epochs5000.pth")
    pinn.plot_solution()
    error_l2 = pinn.compute_l2_error()
    print(f"L2 relative error: {error_l2:.3e}")
    
if __name__ == "__main__":
    main()
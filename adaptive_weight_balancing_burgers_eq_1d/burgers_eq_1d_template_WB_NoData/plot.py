from pinn import PINN
from Test import PINNTester

def main():
    pinn = PINN()
    pinn.load_model()
    pinn.plot_weight_evolution()
    pinn.plot_solution()
    

    error = pinn.compute_l2_error()
    print(f"L2 Error: {error:.6f}")

    
if __name__ == "__main__":
    main()
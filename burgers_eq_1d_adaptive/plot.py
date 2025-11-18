from pinn import PINN

def main():
    pinn = PINN()
    pinn.load_model()
    print(f"N_f: {pinn.N_f} | {pinn.X_f.shape[0]}")
    pinn.plot_solution()
    
if __name__ == "__main__":
    main()
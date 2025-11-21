from pinn import PINN_Burgers

def main():
    pinn = PINN_Burgers(is_adaptive=True)
    pinn.load_model()
    print(f"N_f: {pinn.N_f} | {pinn.X_f.shape[0]}")
    pinn.plot_solution()
    
if __name__ == "__main__":
    main()
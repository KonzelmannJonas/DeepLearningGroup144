from pinn import PINN

def main():
    pinn = PINN(is_adaptive=True)
    pinn.load_model()
    print(f"N_f: {pinn.N_f} | {pinn.X_f.shape[0]}")
    pinn.plot_solution()
    error = pinn.compute_l2_error()
    print(error)
    
if __name__ == "__main__":
    main()
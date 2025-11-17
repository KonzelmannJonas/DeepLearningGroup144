from pinn import PINN

def main():
    pinn = PINN()
    pinn.load_model()
    pinn.plot_solution()
    
if __name__ == "__main__":
    main()
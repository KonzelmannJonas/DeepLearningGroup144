from pinn import PINN

def main():
    pinn = PINN()
    pinn.load_model()
    pinn.save_plot_parameters()
    
if __name__ == "__main__":
    main()
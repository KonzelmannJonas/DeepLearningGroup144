from pinn_model import PINN

def main():
    pinn = PINN()
    pinn.load_model()
    pinn.plot_prediction()

if __name__ == "__main__":
    main()
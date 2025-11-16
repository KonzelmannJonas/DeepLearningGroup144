from pinn_model import PINN

def main():
    pinn = PINN()
    pinn.train()
    pinn.save_model()
    
if __name__ == "__main__":
    main()
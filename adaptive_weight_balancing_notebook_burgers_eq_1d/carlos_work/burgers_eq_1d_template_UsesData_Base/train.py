from pinn import PINN

def main():
    pinn = PINN(epochs=5000)
    pinn.train(method="gradient_balanced")
    pinn.save_model()
    
if __name__ == "__main__":
    main()
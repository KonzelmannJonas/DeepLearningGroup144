from pinn_KAN import PINN

def main():
    pinn = PINN()
    loss_history = pinn.train()
    # save_model in pinn_KAN already stores weights and loss history
    pinn.save_model()
    print("Training finished — model and loss history should be in ./saved_models/")

if __name__ == "__main__":
    main()
import numpy as np
import os
import argparse
# Canvia 'main' pel nom del teu fitxer si es diu 'pinn_KAN.py'
from pinn_KAN import PINN

def main():
    parser = argparse.ArgumentParser(description='Plotting PINN KAN')
    parser.add_argument('--name', type=str, default='experiment', help='Nom de l\'experiment a carregar')
    args = parser.parse_args()

    # 1. Inicialitzem (els epochs no importen per pintar)
    pinn = PINN()

    # Construïm els noms dels fitxers basats en el nom de l'experiment
    model_filename = f"{args.name}_model.pth"
    loss_filename = f"loss_{args.name}.npy" # Format definit a main.py
    
    plot_sol_name = f"{args.name}_prediction.png"
    plot_loss_name = f"{args.name}_loss.png"

    # 2. Carregar Model
    model_path = os.path.join('saved_models', model_filename)
    
    if os.path.exists(model_path):
        print(f"Carregant model de: {model_path}")
        pinn.load_model(model_path)
        
        # Generar plot de la solució
        print("Generant gràfica de la solució...")
        pinn.plot_solution(name=plot_sol_name)
    else:
        print(f"Error: No s'ha trobat el model a {model_path}")
        return

    # 3. Carregar i pintar Historial de Pèrdues (Loss)
    loss_path = os.path.join('saved_models', loss_filename)
    
    if os.path.exists(loss_path):
        print(f"Carregant historial de: {loss_path}")
        loss_history = np.load(loss_path)
        pinn.plot_loss_history(loss_history, name=plot_loss_name)
    else:
        print(f"Avís: No s'ha trobat l'historial de loss a {loss_path}")

    print("Procés de plotting finalitzat.")

if __name__ == "__main__":
    main()
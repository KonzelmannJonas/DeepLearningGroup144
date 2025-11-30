import argparse
# Canvia 'main' pel nom del teu fitxer si es diu 'pinn_KAN.py'
from pinn_KAN import PINN 

def main():
    # 1. Configurem argparse per poder canviar paràmetres des del terminal
    parser = argparse.ArgumentParser(description='Entrenar PINN KAN')
    parser.add_argument('--adam', type=int, default=2000, help='Epochs for Adam')
    parser.add_argument('--lbfgs', type=int, default=3000, help='Max iter for LBFGS')
    parser.add_argument('--name', type=str, default='experiment', help='Nom per guardar fitxers')
    args = parser.parse_args()

    print(f"--- Iniciant entrenament: {args.name} ---")

    # 2. Inicialitzem la PINN passant els arguments
    pinn = PINN(adam_epochs=args.adam, lbfgs_epochs=args.lbfgs)

    # 3. Entrenem (ara retorna l'historial i el temps)
    loss_history, train_time = pinn.train()

    # 4. Calculem l'error final (L2 Error)
    l2_error = pinn.compute_l2_error()
    print(f"Final L2 Error: {l2_error:.5e}")

    # 5. Guardem resultats i model
    # Guardem les mètriques al fitxer de text global (vital per l'HPC)
    pinn.save_metrics(l2_error, train_time, run_name=args.name)
    
    # Guardem el model i l'historial usant el nom proporcionat
    pinn.save_model(loss_history=loss_history, name=f"{args.name}_model.pth")

    print("Entrenament finalitzat i guardat.")

if __name__ == "__main__":
    main()
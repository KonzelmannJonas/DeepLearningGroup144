#!/bin/sh
#BSUB -q gpuv100
#BSUB -J PINN_full_5k
#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -W 02:00  # 2 hours for 5000 epochs
#BSUB -o output_%J.txt
#BSUB -e error_%J.txt

# Load modules
module load cuda/12.4.1
module load python3/3.12.11

echo "=== Starting PINN with Data Loss and Weight Balancing ==="
echo "Training for 5000 epochs"
python3 train_full.py

echo "=== Job completed ==="

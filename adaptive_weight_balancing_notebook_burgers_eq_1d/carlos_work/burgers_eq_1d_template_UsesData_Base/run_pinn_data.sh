#!/bin/sh
#BSUB -q gpuv100
#BSUB -J PINN_data
#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -W 00:10
#BSUB -o output.txt
#BSUB -e error.txt

# Load modules
module load cuda/12.4.1
module load python3/3.12.11

# Run directly
echo "=== Starting PINN with Data Loss ==="
python3 train_with_data.py

echo "=== Job completed ==="

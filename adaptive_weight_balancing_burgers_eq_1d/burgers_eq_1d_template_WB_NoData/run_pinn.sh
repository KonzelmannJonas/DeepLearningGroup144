#!/bin/sh
#BSUB -q gpuv100
#BSUB -J PINN_test
#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -W 00:10  # Only 10 minutes for testing
#BSUB -o output.txt
#BSUB -e error.txt

# Load modules
module load cuda/12.4.1
module load python3/3.12.11

# Run directly with system python (no virtual env)
echo "=== Direct execution ==="
python3 train_pinn.py

echo "=== Job completed ==="

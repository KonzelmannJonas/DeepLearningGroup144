#!/bin/sh
### --- Job Configuration ---
#BSUB -q gpuv100             # V100 GPU queue (very fast)
#BSUB -J PINN_KAN_Oriol      # Job Name seen in the queue
#BSUB -n 4                   # Request 4 CPU cores
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 exclusive GPU
#BSUB -W 00:20               # Max runtime (4 hours)
#BSUB -R "rusage[mem=16GB]"  # Request 16GB of RAM
### --- Log Files ---
#BSUB -o logs/output_%J.txt  # Standard output log
#BSUB -e logs/error_%J.txt   # Error log

### 1. Load System Modules
# Using the specific versions compatible with your setup
module load cuda/12.4.1
module load python3/3.12.11

### 2. Activate Virtual Environment
# IMPORTANT FIX: Using ".." to look for 'venv' in the parent directory
source ../venv/bin/activate

### 3. Safety Checks
echo "Running on host: $(hostname)"
echo "GPU available:"
nvidia-smi

### 4. RUN TRAINING
# Executing the script with the defined arguments
python train_KAN.py --adam 1000 --lbfgs 4000 --name "Run_Final_KAN"


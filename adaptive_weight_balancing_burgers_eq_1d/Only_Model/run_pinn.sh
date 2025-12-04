#!/bin/sh
### --- FAST TRACK Configuration ---
#BSUB -q gpuv100        # KEY CHANGE: Using A100 queue (much less crowded)
#BSUB -J PINN_WB_DATA       # Job Name
#BSUB -n 4                   # 4 CPU cores
#BSUB -gpu "num=1:mode=exclusive_process" # 1 GPU
#BSUB -W 00:25               # KEY CHANGE: Only 25 minutes (max priority)
#BSUB -R "rusage[mem=8GB]"   # Requesting less RAM to facilitate entry
### --- Logs ---
#BSUB -o logs/output_%J.txt
#BSUB -e logs/error_%J.txt

### 1. Load Modules
module load cuda/12.4.1
module load python3/3.12.11

### 2. Activate Environment
source /dtu/blackhole/08/221256/WB/WB_env

### 3. Safety Check
echo "Running on host: $(hostname)"
nvidia-smi

### 4. EXECUTE (Short version for testing)
# Reducing epochs to finish in less than 15 minutes
python3 train_pinn.py

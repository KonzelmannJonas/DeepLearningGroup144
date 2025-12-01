#!/bin/sh
### --- FAST TRACK Configuration ---
#BSUB -q gpuv100             # KEY CHANGE
#BSUB -J PINN_adaptive       # Job Name
#BSUB -n 4                   # 4 CPU cores
#BSUB -gpu "num=1:mode=exclusive_process" # 1 GPU
#BSUB -W 00:15               # KEY CHANGE: Only 15 minutes (max priority)
#BSUB -R "rusage[mem=8GB]"   # Requesting less RAM to facilitate entry
### --- Logs ---
#BSUB -o logs/output_%J.txt
#BSUB -e logs/error_%J.txt

### 1. Load Modules
module load cuda/12.4.1
module load python3/3.12.11

### 2. Activate Environment
source /dtu/blackhole/1d/223453/venv/bin/activate

### 3. Safety Check
echo "Running on host: $(hostname)"
nvidia-smi

### 4. EXECUTE (Short version for testing)
# Reducing epochs to finish in less than 15 minutes
python3 train.py
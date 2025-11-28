# ==============================================================================
# 📢 QUICK GUIDE FOR THE TEAM (GITHUB/RESULTS)
# ==============================================================================
# How to compare results correctly:
#
# 1. Loss & Accuracy:
#    -> Must be IDENTICAL (or differ only at the 5th decimal digit).
#    -> Why: The script forces the same math precision and random seeds.
#
# 2. Training Time:
#    -> Do NOT compare raw seconds.
#    -> Compare NORMALIZED TIME.
#
#    [FORMULA]: Normalized Time = Real Time (seconds) * Performance Index
#
#    -> Interpretation: If your Normalized Time is LOWER than your colleague's,
#       your code is MORE EFFICIENT, even if your GPU is slower.
# ==============================================================================
 
import torch
import random
import numpy as np
import os
import time

def setup_unified_environment(seed_val=42):
    """
    Configures the environment to ensure MAXIMUM reproducibility across different GPUs.
    Returns the device (cuda/cpu) and forces deterministic algorithms.
    """
    print(f"\n╔════════ UNIFIED ENVIRONMENT SETUP (Seed: {seed_val}) ════════╗")

    # 1. Set SEEDS (Lock randomness)
    # This ensures that random initialization of weights is identical for everyone.
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 2. Hardware Configuration "Lowest Common Denominator"
    if torch.cuda.is_available():
        # A. DISABLE TensorFloat-32 (TF32)
        # Crucial: RTX 30xx/40xx series use TF32 by default (faster but less precise).
        # GTX 10xx/20xx do not support it. 
        # By disabling it, we force newer GPUs to be mathematically identical to older ones.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # B. FORCE DETERMINISM
        # Ensures CuDNN uses reproducible convolution algorithms, 
        # sacrificing a bit of speed for consistency.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # C. Workspace Configuration (needed for some deterministic operations)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        device_name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        print(f"║ ✅ GPU Detected: {device_name} ({vram} GB)")
        print(f"║ ⚖️  Mode: Deterministic (TF32 Disabled, Float32 Forced)")
    else:
        print("║ ⚠️  WARNING: No GPU detected. Using CPU.")
        device_name = "CPU"
    
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gpu_performance_index():
    """
    Runs a standard benchmark to assign a performance score to the current GPU.
    Use this score to normalize training times when comparing results.
    """
    if not torch.cuda.is_available():
        return 1.0 # Base score for CPU

    print("⏳ Calculating GPU performance index...", end="\r")
    
    # Create heavy matrices for testing
    size = 4000
    x = torch.randn(size, size, device="cuda")
    y = torch.randn(size, size, device="cuda")
    
    # Warmup (heat up the GPU to reach operating frequency)
    for _ in range(5):
        _ = torch.mm(x, y)
    
    # Precise Timing
    torch.cuda.synchronize() # Wait for GPU to finish previous tasks
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    _ = torch.mm(x, y) # Test operation
    end_event.record()
    
    torch.cuda.synchronize() # Wait for GPU to finish the test
    
    elapsed_ms = start_event.elapsed_time(end_event)
    
    # Arbitrary Index: 10000 / ms. 
    # Example: If it takes 100ms -> Score 100. If 50ms -> Score 200.
    # Higher score = More powerful GPU.
    score = round(10000 / elapsed_ms, 2)
    
    print(f"🚀 Hardware Performance Index: {score} (Use this to normalize times)\n")
    return score
import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Fixes random seeds across Python, NumPy, PyTorch, and CUDA/cuDNN environments
    to guarantee 100% experiment reproducibility.

    Args:
        seed (int): The seed value to set. Default: 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Ensures multi-GPU setup reproducibility
    
    # Force cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Reproducibility] Random seed fixed to: {seed}")


def count_parameters(model):
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6

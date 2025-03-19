import torch
import gc
import os
import numpy as np

def create_dummy_model(hidden_dim=128, num_layers=2, vocab_size=10000):
    """Create a small model for testing"""
    from model import SSMTransformerModel
    
    print("Creating test model...")
    model = SSMTransformerModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=4,
        state_dim=8,
        max_seq_len=1024
    )
    return model

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    """Clear memory and cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_dir(dir_path):
    """Ensure directory exists"""
    os.makedirs(dir_path, exist_ok=True)
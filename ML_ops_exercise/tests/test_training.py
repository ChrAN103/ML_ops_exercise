import sys
import os

import torch
# Add the source directory to sys.path relative to this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/ml_ops_ex")))
from train import train

def test_training():
    """Test the training function runs without errors."""
    try:
        train(lr=1e-3, batch_size=32, epochs=1,save=False)
    except Exception as e:
        assert False, f"Training function raised an exception: {e}"

if __name__ == "__main__":
    test_training()
    print("Training test passed!")
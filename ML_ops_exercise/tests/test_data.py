import torch
from torch.utils.data import Dataset
import os
import sys

# Add src to path so we can import ml_ops_ex
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tests import _PATH_DATA
# from _PATH_DATA import raw # This is invalid because _PATH_DATA is a string path, not a module
from ml_ops_ex.data import corrupt_mnist as raw

# def test_my_dataset():
#     """Test the MyDataset class."""
#     dataset = MyDataset("data/raw")
#     assert isinstance(dataset, Dataset)

def test_data():
    train,test = raw()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train,test]:
        for x,y in dataset:
            assert x.shape in [(1,28,28),(784,)]
            assert 0 <= y <= 9
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(10)).all()
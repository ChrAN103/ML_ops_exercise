import torch
from ml_ops_ex.data import corrupt_mnist as raw

# def test_my_dataset():
#     """Test the MyDataset class."""
#     dataset = MyDataset("data/raw")
#     assert isinstance(dataset, Dataset)


def test_data():
    train, test = raw()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape in [(1, 28, 28), (784,)]
            assert 0 <= y <= 9
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(10)).all()

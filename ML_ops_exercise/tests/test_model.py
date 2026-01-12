import os
import pytest
import torch
from tests import _PATH_DATA
from ml_ops_ex.model import Model

# Conditional skip if data files are not found
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_something_about_data():
    ...

def test_model():
    """Test the Model class."""
    model = Model()  
    try:
        input_tensor = torch.randn(1, 1, 28, 28)
        output_tensor = model(input_tensor)
    except Exception as e:
        assert False, f"Model forward pass raised an exception: {e}" # either input not 4D tensor or input shape wrong != (N,1,28,28)
    assert output_tensor.shape == (1, 10), f"Expected output shape (1, 10), got {output_tensor.shape}"

@pytest.mark.parametrize("batch_size", [32,64])
def test_parametrized_examples(batch_size):
    """Parametrized test example."""
    model = Model()
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {output_tensor.shape}"

def test_error_on_wrong_shape():
    """Test that the model raises an error for wrong input dimensions."""
    model = Model()
    with pytest.raises(ValueError, match="Expected input to be 4D tensor"):
        model(torch.randn(1, 28, 28))  # Invalid shape
    with pytest.raises(ValueError, match=r"Expected input shape to be \(N, 1, 28, 28\)"):  # invalid channel size
        model(torch.randn(1, 3, 28, 28))  # Invalid channel size
    with pytest.raises(ValueError, match=r"Expected input shape to be \(N, 1, 28, 28\)"):  # invalid rows
        model(torch.randn(1, 1, 28, 29))  # Invalid rows
    with pytest.raises(ValueError, match=r"Expected input shape to be \(N, 1, 28, 28\)"):  # invalid coloumns
        model(torch.randn(1, 1, 27, 28))  # Invalid coloumns

if __name__ == "__main__":
    model = Model()
    test_model()
    print("Model test passed!")
    test_parametrized_examples(batch_size=32)
    print("Parametrized test passed!")
    # test_error_on_wrong_shape()
    # print("Wrong input dimension test passed!")

from ml_ops_ex.train import train


def test_training():
    """Test the training function runs without errors."""
    try:
        train(lr=1e-3, batch_size=32, epochs=1, save=False)
    except Exception as e:
        assert False, f"Training function raised an exception: {e}"


if __name__ == "__main__":
    test_training()
    print("Training test passed!")

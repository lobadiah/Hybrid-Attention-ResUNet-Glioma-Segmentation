import torch

# Assuming these are the classes for the five models to be tested
from models import Model1, Model2, Model3, Model4, Model5

def test_model(model_class, input_tensor):
    model = model_class()
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"{model_class.__name__} output shape: {output.shape}")

def main():
    # Create a dummy input tensor with an appropriate size
    input_tensor = torch.randn(1, 3, 256, 256)  # Adjust the shape according to your model requirements

    # Test all five models
    test_model(Model1, input_tensor)
    test_model(Model2, input_tensor)
    test_model(Model3, input_tensor)
    test_model(Model4, input_tensor)
    test_model(Model5, input_tensor)

if __name__ == "__main__":
    main()
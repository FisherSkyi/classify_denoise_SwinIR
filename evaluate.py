import torch
from torchvision.models import resnet18
from train_cnn_csv import SimpleCNN  # Assuming you have a SimpleCNN class defined in simple_cnn.py
import torch.nn as nn
import load
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def test(model_path: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_path.startswith("resnet18"):
        # If using ResNet18, we need to modify the final layer for 43 classes
        model = resnet18() # initializing resnet18 with random weights
        model.fc = nn.Linear(model.fc.in_features, 43)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        test_loader = load.test_load()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Test accuracy: {correct / total:.4f}")

    elif model_path.startswith("cnn"):
        model = SimpleCNN(num_classes=43)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        test_loader = load.test_load()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Test accuracy: {correct / total:.4f}")


def main():
    model_path = ""  # Path to the saved model
    test(model_path)

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import load_dirty  # assumes train_load() is defined in load.py
from torchsummary import summary
import argparse
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()
lr = args.lr

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# CSV file setup
csv_file = 'cnn_dirty.csv'
file_exists = os.path.isfile(csv_file)

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# Main training loop
def main():
    print(torch.cuda.get_device_name(0))

    # Model, criterion, optimizer
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training loop
    train_loader, val_loader = load_dirty.train_load()

    num_epochs = args.epochs

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Accuracy', 'Val_Loss', 'Val_Accuracy'])


    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        # Save stats to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}"])

        print(f"Epoch {epoch+1}/10")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), f"cnn_dirty_Epoch{args.epoch}.pth")
    # print(summary(SimpleCNN(), input_size=(3, 64, 64)))

if __name__ == "__main__":
    main()

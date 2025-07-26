import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import load
import argparse
import csv
import os
# from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()
lr = args.lr

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# CSV file setup
csv_file = 'resnet_clean.csv'
file_exists = os.path.isfile(csv_file)

def train(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Validating", leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def main():
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.current_device())
    # print(torch.cuda.device(0))
    # print(torch.cuda.get_device_name(0))

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 43)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = load.train_load()
    num_epochs = args.epochs  # Use command-line argument for epochs

    # Write CSV header if file doesn't exist
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

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    torch.save(model.state_dict(), f"resnet18_clean_Epoch{args.epochs}_lr{args.lr}.pth")
    # print(summary(model, input_size=(3, 64, 64)))

if __name__ == "__main__":
    main()

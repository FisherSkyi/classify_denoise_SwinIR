from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import os
import pandas as pd
from PIL import Image


def train_load():
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Resizes the image to 64*64 pixels
        transforms.ToTensor(),        # Converts PIL Image or NumPy ndarray to a PyTorch Tensor
        # note that channels are reordered, HWC → CHW
        # add normalization that matches the ImageNet training data.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    # Load entire dataset
    full_dataset = datasets.ImageFolder(root='GTSRB', transform=transform)

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader

class GTSRBTestDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        image = Image.open(img_path)
        # Crop ROI using bounding box
        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        image = image.crop((x1, y1, x2, y2))
        if self.transform:
            image = self.transform(image)
        label = row['ClassId']
        return image, label

def test_load():
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    test_dataset = GTSRBTestDataset(
        img_dir='GTSRB/test/Images',
        csv_path='GTSRB/test/test_labels.csv',
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader
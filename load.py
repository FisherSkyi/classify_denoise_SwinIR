from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import os
import pandas as pd
from PIL import Image

class GTSRBTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (image_path, label)
        self.class_to_idx = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted(entry.name for entry in os.scandir(self.root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # convert ensures 3 channels

        if self.transform:
            image = self.transform(image)

        return image, label

def train_load():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    full_dataset = GTSRBTrainDataset(
        root_dir='GTSRB/train',
        transform=transform
    )
    # print(f"Train size: {len(full_dataset)}")
    train_size = int(0.8 * len(full_dataset))
    # print(f"Train size: {train_size}")
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=False)

    return train_loader, val_loader


class GTSRBTestDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path, sep=';')
        self.transform = transform

    def __len__(self):
        return len(self.data)

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
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = GTSRBTestDataset(
        img_dir='GTSRB/test-gaussian-20.0',
        csv_path='GTSRB/GT-final_test.csv',
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=False)

    return test_loader

# def main():
#     train_loader, val_loader = train_load()
#     test_loader = test_load()
#
#     print(f"Train Loader: {len(train_loader.dataset)} samples")
#     print(f"Validation Loader: {len(val_loader.dataset)} samples")
#     print(f"Test Loader: {len(test_loader.dataset)} samples")
#
# if __name__ == "__main__":
#     main()

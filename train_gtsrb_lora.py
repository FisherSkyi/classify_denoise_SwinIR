import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import load
import LoRA.loralib as lora
from SwinIR.models.network_swinir import SwinIR

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

# --- Model ---
class SwinIRClassifier(nn.Module):
    def __init__(self, swinir_model, num_classes=43):
        super().__init__()
        self.backbone = swinir_model
        self.feature_dim = swinir_model.embed_dim
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool along the sequence length
        self.classifier = lora.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # This forward pass correctly extracts deep features from the SwinIR backbone.
        # Use the backbone's own padding and normalization logic
        if x.shape[-1] != 128 or x.shape[-2] != 128:
            x = torch.nn.functional.interpolate(x, size=(128, 128),
                                                mode='bilinear', align_corners=False)
        x = self.backbone.check_image_size(x)
        self.backbone.mean = self.backbone.mean.type_as(x)
        x = (x - self.backbone.mean) * self.backbone.img_range

        # 1. Shallow feature extraction: projects from 3 channels to embed_dim (96)
        x = self.backbone.conv_first(x)

        # 2. Deep feature extraction: pass through the transformer blocks
        # This part mimics the backbone's 'forward_features' method to get the final feature map.
        x_size = (x.shape[2], x.shape[3])
        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        for layer in self.backbone.layers:
            x = layer(x, x_size)

        features = self.backbone.norm(x)  # Shape: [B, L, C], where C is 96. This is our feature vector.

        # 3. Pool features and classify
        features = features.permute(0, 2, 1)  # -> [B, C, L]
        pooled_features = self.pool(features)  # -> [B, C, 1]
        pooled_features = torch.flatten(pooled_features, 1)  # -> [B, C]

        return self.classifier(pooled_features)

backbone = SwinIR(
    img_size=128,
    patch_size=1,
    in_chans=3,
    embed_dim=180,
    depths=[6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6],
    window_size=8,
    mlp_ratio=2,
    img_range=1.0,
    upsampler='', # Use '' for denoising models
    upscale=1,
    resi_connection='1conv'
)

# NOTE: SwinIR pretrained models often store weights under a 'params' key.
# Load the state dictionary.
pretrained_model = torch.load('SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth')
if 'params' in pretrained_model:
    pretrained_state_dict = pretrained_model['params']
else:
    pretrained_state_dict = pretrained_model

backbone.load_state_dict(pretrained_state_dict, strict=False)

# backbone.load_state_dict(pretrained_model, strict=False)

model = SwinIRClassifier(backbone, num_classes=43)
lora.mark_only_lora_as_trainable(model)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# def pad_to_window_size(img, window_size):
#     _, _, h, w = img.shape
#     pad_h = (window_size - h % window_size) % window_size
#     pad_w = (window_size - w % window_size) % window_size
#     return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

def train(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # inputs = pad_to_window_size(inputs, window_size=8)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main():
    train_loader, val_loader = load.train_load()
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    torch.save(lora.lora_state_dict(model), 'swinir_gtsrb_lora.pth')


if __name__ == "__main__":
    main()

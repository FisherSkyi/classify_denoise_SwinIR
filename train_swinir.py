import os, math, random, argparse, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim


from SwinIR.models.network_swinir import SwinIR           # pip install timm && clone SwinIR

# ------------------------------------------------------------------------------
# 1.  Dataset that returns (noisy, clean)
# ------------------------------------------------------------------------------
class GTSRBDenoiseDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 img_size: int = 128,
                 noise_std: float = 15 / 255.0):
        self.paths = [str(p) for p in Path(root_dir).rglob("*.*")
                      if p.suffix.lower() in {".png", ".ppm", ".jpg", ".jpeg"}]
        self.noise_std = noise_std
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()  # 0‒1 range
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        clean = self.tf(img)
        # pixel-wise Gaussian
        # [C,H,W], float32
        noisy = (clean + torch.randn_like(clean) *
                 self.noise_std).clamp(0., 1.)
        return noisy, clean


# ------------------------------------------------------------------------------
# 2.  PSNR helper
# ------------------------------------------------------------------------------
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ------------------------------------------------------------------------------
# 3.  Train / Val loops
# ------------------------------------------------------------------------------


def run_epoch(model, loader, criterion):
    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-4,  # typical for image-denoising with SwinIR
        betas=(0.9, 0.999),
        weight_decay=0  # Adam usually works best with 0 or a tiny WD for vision
    )
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss, running_psnr = 0., 0.

    with torch.set_grad_enabled(is_train):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)

            loss = criterion(output, clean)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * noisy.size(0)
            running_psnr += psnr(output.detach(), clean).item() * noisy.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_psnr / n


# ------------------------------------------------------------------------------
# 4.  Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="GTSRB/train", help="path with clean images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--noise", type=int, default=15, choices=[15, 25, 50])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="swinir_gtsrb_dn.pth")
    args = parser.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))

    #  Dataset & loaders -------------------------------------------------------
    full_set = GTSRBDenoiseDataset(args.root, args.img_size, args.noise / 255.)
    val_split = int(0.2 * len(full_set))
    train_set, val_set = torch.utils.data.random_split(
        full_set, [len(full_set) - val_split, val_split],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch,
                            shuffle=False, num_workers=2, pin_memory=True)

    #  SwinIR (color denoising variant) ----------------------------------------
    model = SwinIR(upscale=1, in_chans=3, img_size=args.img_size,
                   window_size=8, img_range=1.,
                   depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    model.to(device)

    criterion = nn.L1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #  Training ---------------------------------------------------------------
    best_psnr = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_psnr = run_epoch(model, train_loader, criterion)
        vl_loss, vl_psnr = run_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch:2d}/{args.epochs} "
              f"| Train L1 {tr_loss:.4f}  PSNR {tr_psnr:.2f}dB "
              f"| Val L1 {vl_loss:.4f}  PSNR {vl_psnr:.2f}dB "
              f"| {time.time()-t0:.1f}s")

        # save best
        if vl_psnr > best_psnr:
            best_psnr = vl_psnr
            torch.save(model.state_dict(), args.save)
            print(f" :) Saved checkpoint to {args.save}")

    print(f"Finished. Best Val PSNR = {best_psnr:.2f} dB")
# wand is a library used for image processing tasks，
# e.g. open and process image files.
# from wand.image import Image

import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def add_gaussian_noise(img: Image.Image, std: float) -> Image.Image:
    """
    Return a copy of *img* with additive Gaussian noise.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, std, rgb.shape)
    noisy = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def process_directory(src_root: Path, dst_root: Path, std: float) -> None:
    """
    Walk through *src_root* and store a noisy version of every image
    under the mirrored directory structure rooted at *dst_root*.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

    for class_dir in src_root.iterdir():
        if class_dir.name == ".DS_Store" or not class_dir.is_dir():
            continue

        # create a new Path object that represents a subdirectory path.
        dst_class_dir = dst_root / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing class: {class_dir.name}")

        for file in class_dir.iterdir():
            if file.suffix.lower() not in exts or not file.is_file():
                continue

            dst_file = dst_class_dir / file.name
            try:
                with Image.open(file) as im:
                    noisy = add_gaussian_noise(im, std)
                    noisy.save(dst_file)
            except (OSError, ValueError) as err:
                print(f"  ! Skipped {file.name}: {err}")


def main() -> None:
    try:
        std = float(input("noise level (standard deviation): ").strip())
    except ValueError:
        print("Please enter a valid floating-point number.")
        return

#src_root = Path("data/filtered-tsrd-test")          # original clean dataset
#dst_root = Path(f"data/filtered-tsrd-test{std}")    # noisy copy destination
    src_root = Path("data/tsrd-train")          # original clean dataset
    dst_root = Path(f"data/tsrd-train{std}")    # noisy copy destination

    if not src_root.exists():
        print(f"Source folder not found: {src_root}")
        return

    # Create destination folder (and parents) if needed
    dst_root.mkdir(parents=True, exist_ok=True)

    process_directory(src_root, dst_root, std)
    print("Done.")


if __name__ == "__main__":
    main()

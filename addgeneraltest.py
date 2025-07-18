import os
from pathlib import Path
import numpy as np
from PIL import Image
from wand.image import Image as WandImage
import io


def add_gaussian_noise(img: Image.Image, std: float) -> Image.Image:
    """
    Return a copy of *img* with additive Gaussian noise.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, std, rgb.shape)
    noisy = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def add_impulse_noise(img: Image.Image, attenuate: float) -> Image.Image:
    """
    Return a copy of *img* with impulse noise.
    """
    # Convert PIL image to a buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Create Wand image from buffer
    with WandImage(blob=buffer.getvalue()) as wand_img:
        wand_img.noise("impulse", attenuate=attenuate)
        # Convert back to PIL image
        return Image.open(io.BytesIO(wand_img.make_blob('png')))


def add_poisson_noise(img: Image.Image, attenuate: float) -> Image.Image:
    """
    Return a copy of *img* with Poisson noise.
    """
    # Convert PIL image to a buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Create Wand image from buffer
    with WandImage(blob=buffer.getvalue()) as wand_img:
        wand_img.noise("poisson", attenuate=attenuate)
        # Convert back to PIL image
        return Image.open(io.BytesIO(wand_img.make_blob('png')))


def process_directory(src_root: Path, dst_root: Path, noise_type: str, severity: float) -> None:
    """
    Walk through *src_root* and store a noisy version of every image
    under the mirrored directory structure rooted at *dst_root*.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

    for class_dir in src_root.iterdir():
        if class_dir.name == ".DS_Store" or not class_dir.is_dir():
            continue

        dst_class_dir = dst_root / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing class: {class_dir.name}")

        for file in class_dir.iterdir():
            if file.suffix.lower() not in exts or not file.is_file():
                continue

            dst_file = dst_class_dir / file.name
            try:
                with Image.open(file) as im:
                    if noise_type.lower() == "gaussian": # convert to lower cases
                        noisy = add_gaussian_noise(im, severity)
                        noisy.save(dst_file)
                    else:
                        noise_func = add_impulse_noise if noise_type.lower() == "impulse" else add_poisson_noise
                        noisy = noise_func(im, severity)
                        noisy.save(dst_file)
            except (OSError, ValueError) as err:
                print(f"  ! Skipped {file.name}: {err}")


def main() -> None:
    # Get noise type
    noise_type = input("Enter noise type (gaussian, impulse, poisson): ").strip().lower()
    if noise_type not in ["gaussian", "impulse", "poisson"]:
        print("Invalid noise type. Choose 'gaussian', 'impulse', or 'poisson'.")
        return

    # Get severity (standard deviation for Gaussian, attenuate for others)
    try:
        severity = float(
            input("Enter noise severity (standard deviation for Gaussian, attenuate for others): ").strip())
    except ValueError:
        print("Please enter a valid floating-point number.")
        return

    src_root = Path("data/tsrd-test")  # original clean dataset
    dst_root = Path(f"data/tsrd-test-{noise_type}-{severity}")  # noisy copy destination

    if not src_root.exists():
        print(f"Source folder not found: {src_root}")
        return

    # Create destination folder (and parents) if needed
    dst_root.mkdir(parents=True, exist_ok=True)

    process_directory(src_root, dst_root, noise_type, severity)
    print("Done.")


if __name__ == "__main__":
    main()

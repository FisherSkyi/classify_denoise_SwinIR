import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw

def add_circle(img: Image.Image, size: float, max_size: float) -> Image.Image:
    """
    Add a random circle to the image with size relative to image dimensions.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Determine circle radius (between 1/10 and max_size of the smaller dimension)
    min_dim = min(width, height)
    max_radius = int(min_dim * max_size / 2)
    min_radius = int(min_dim * 0.1 / 2)
    radius = random.randint(min_radius, max_radius)
    
    # Random position ensuring the circle stays within bounds
    x = random.randint(radius, width - radius)
    y = random.randint(radius, height - radius)
    
    # Random color (RGB)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Draw filled circle
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return img

def add_triangle(img: Image.Image, size: float, max_size: float) -> Image.Image:
    """
    Add a random triangle to the image with size relative to image dimensions.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Determine triangle size (between 1/10 and max_size of the smaller dimension)
    min_dim = min(width, height)
    max_side = int(min_dim * max_size)
    min_side = int(min_dim * 0.1)
    side = random.randint(min_side, max_side)
    
    # Random position for the triangle's centroid
    x = random.randint(side // 2, width - side // 2)
    y = random.randint(side // 2, height - side // 2)
    
    # Define triangle vertices (equilateral triangle approximation)
    h = int(side * (3 ** 0.5) / 2)  # Height of equilateral triangle
    points = [
        (x, y - h // 2),  # Top
        (x - side // 2, y + h // 2),  # Bottom-left
        (x + side // 2, y + h // 2)   # Bottom-right
    ]
    
    # Random color (RGB)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Draw filled triangle
    draw.polygon(points, fill=color)
    return img

def add_square(img: Image.Image, size: float, max_size: float) -> Image.Image:
    """
    Add a random square to the image with size relative to image dimensions.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Determine square size (between 1/10 and max_size of the smaller dimension)
    min_dim = min(width, height)
    max_side = int(min_dim * max_size)
    min_side = int(min_dim * 0.1)
    side = random.randint(min_side, max_side)
    
    # Random position ensuring the square stays within bounds
    x = random.randint(side // 2, width - side // 2)
    y = random.randint(side // 2, height - side // 2)
    
    # Random color (RGB)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Draw filled square
    draw.rectangle([x - side // 2, y - side // 2, x + side // 2, y + side // 2], fill=color)
    return img

def add_random_shapes(img: Image.Image, max_size: float) -> Image.Image:
    """
    Add one circle, one triangle, and one square to the image with random sizes, positions, and colors.
    """
    # List of shape functions
    shape_functions = [add_circle, add_triangle, add_square]
    
    # Shuffle the order of shapes to apply them in random order
    random.shuffle(shape_functions)
    
    # Apply each shape
    for shape_func in shape_functions:
        img = shape_func(img, 0, max_size)
    
    return img

def process_directory(src_root: Path, dst_root: Path, max_size: float) -> None:
    """
    Walk through *src_root* and store a version of each image with shapes
    under the mirrored directory structure rooted at *dst_root*.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".ppm"}
    
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
                    # Convert to RGB if necessary
                    im = im.convert("RGB")
                    # Add all three shapes
                    modified = add_random_shapes(im, max_size)
                    modified.save(dst_file)
            except (OSError, ValueError) as err:
                print(f"  ! Skipped {file.name}: {err}")

def main() -> None:
    # Get maximum shape size
    try:
        max_size = float(input("Enter maximum shape size (as a fraction of image, e.g., 0.25 for 1/4): ").strip())
        if not 0.1 <= max_size <= 0.5:
            print("Please enter a value between 0.1 and 0.5.")
            return
    except ValueError:
        print("Please enter a valid floating-point number.")
        return

    # dest = input("Please input the target directory to add shapes to: ")
    dest = "GTSRB/test"
    src_root = Path(dest)          # Original clean dataset
    print(src_root)
    dst_root = Path(dest + "-" + str(max_size))  # Destination with shapes

    if not src_root.exists():
        print(f"Source folder not found: {src_root}")
        return

    # Create destination folder (and parents) if needed
    dst_root.mkdir(parents=True, exist_ok=True)

    process_directory(src_root, dst_root, max_size)
    print("Done.")

if __name__ == "__main__":
    main()

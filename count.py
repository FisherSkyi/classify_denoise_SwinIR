import os
import glob
import csv

root_dir = './GTSRB/train'  # Change to your dataset root path
output_csv = 'class_image_counts.csv'

# Define allowed image extensions
image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.ppm')

results = []

for class_name in sorted(os.listdir(root_dir)):
    class_path = os.path.join(root_dir, class_name)
    print(class_path)
    if os.path.isdir(class_path):
        count = 0
        for ext in image_extensions:
            count += len(glob.glob(os.path.join(class_path, ext)))
        results.append((class_name, count))

# Save to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'ImageCount'])
    writer.writerows(results)

print(f"Image counts per class saved to {output_csv}")
from wand.image import Image
import os
import numpy as np


dirname = input("input attenuation")
var2 = float(dirname)

# Directory containing subdirectories with images
#directory_path = 'files/testgaussian10'  # Replace with the actual directory path
#directory_path = 'files/testgaussian5'  # Replace with the actual directory path
directory_path = 'files/testgaussian'+dirname  # Replace with the actual directory path

# Iterate through all subdirectories in the directory
for subdir in os.listdir(directory_path):
    subdirectory_path = os.path.join(directory_path, subdir)
    print(subdirectory_path)
    if '.DS_Store' in subdirectory_path:
        print("skip")
        continue    
    # Iterate through all image files in the subdirectory
    for filename in os.listdir(subdirectory_path):
        image_path = os.path.join(subdirectory_path, filename)
        with Image(filename=image_path) as img:
                img.noise("gaussian", attenuate=int(var2))
                img.save(filename=image_path)
        


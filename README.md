# CV_noise
Computer Vision (Denoising)
# How to use this repo:
```bash
#!/bin/bash

input_folder="./filtered-tsrd-test"

for filepath in "$input_folder"/*; do
    filename=$(basename "$filepath")
    prefix=${filename:0:3}

    # Create target directory if it doesn't exist
    mkdir -p "$input_folder/$prefix"

    # Move file
    mv "$filepath" "$input_folder/$prefix/"
done
```
This is the script to select data from raw dataset.
```bash
#!/bin/bash

input_folder="./tsrd-train"
unused_folder="$input_folder/unused"

mkdir -p "$unused_folder"

top_folders=$(find "$input_folder" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    count=$(find "$dir" -type f -iname "*.png" | wc -l)
    echo "$count $dir"
done | sort -nr | head -n 10 | awk '{print $2}')

find "$input_folder" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if ! echo "$top_folders" | grep -Fxq "$dir"; then
        if [ "$dir" != "$unused_folder" ]; then
            mv "$dir" "$unused_folder/"
        fi
    fi
done
```
This is the script to choose the top 10 classes with mose number of samples.
### Day 1 of coding July 17
Helped everyone clone github to their computers

Also started adding noise to the dataset

``print("hello world!\n"");``

>If you research interest is traffic sign recognition or 
detection,please download the database without 
submitting application forms

Finish training model B and C and get results

> 

### Day 2 of coding July 18
Finish research presentation1

Continue reading related work
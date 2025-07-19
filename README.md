# CV_noise
Computer Vision (Denoising)


[Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)

J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Neural Networks, Available online 20 February 2012, ISSN 0893-6080, 10.1016/j.neunet.2012.02.016. (http://www.sciencedirect.com/science/article/pii/S0893608012000457) Keywords: Traffic sign recognition; Machine learning; Convolutional neural networks; Benchmarking



---
# How to use this repo:
This is the script to select data from raw dataset.
```bash
#!/bin/bash

input_folder="./filtered-tsrd-test"

for filepath in "$input_folder"/*; do
    filename=$(basename "$filepath")
    prefix=${filename:0:3}

    # Create target directory if it doesn't exist
    mkdir -p "$input_folder/$prefix"

### Instruction:

Version1:

1. Run 'addgeneral.py' to create a data file "..." with induced noise
2. Rename ...
3. Run 'cleanmodel.py' to create and train clean model
4. Run 'noisemodel.py' to create and train noise model 


    # Move file
    mv "$filepath" "$input_folder/$prefix/"
done
```
---
This is the script to choose the top 10 classes with mose number of samples.
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
---
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
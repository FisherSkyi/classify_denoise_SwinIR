## License

This repository is licensed under the [Apache License 2.0](LICENSE).  
It includes code from [SwinIR](https://github.com/JingyunLiang/SwinIR),  
which is also licensed under the Apache License 2.0.

Dataset: [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)\
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011. 




```text
# Directory structure of the dataset
test/
  ├── Images/
  │    ├── 00000.ppm
  │    ├── 00001.ppm
  │    └── ...
  └── GT-final_test.csv  # contains image names and labels
train/
  ├── class0/
  │    ├── img1.png
  │    ├── ...
  │    └── GT-00000.csv
  ├── class1/
  │    ├── imgA.png
  │    ├── ...
  ├── Readme-Images.txt
```

# How to use this repo:


### Instruction:

Version1:

1. Run 'addgeneral.py' to create a data file "..." with induced noise
2. Rename ...
3. Run 'cleanmodel.py' to create and train clean model
4. Run 'noisemodel.py' to create and train noise model 

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
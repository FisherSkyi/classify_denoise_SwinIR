# Road Sign Classification with Denoising Pipeline Approach
[Yu Letian](https://github.com/FisherSkyi), Sun Yuqi, Zhang Jiaqi

[email of author](yuletian@u.nus.edu)

---

This repository is the PyTorch implementation of our idea: *Road Sign Classification with Denoising Pipeline Approach*

#### Contents
1. [Dataset](#Dataset)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [License and Acknowledgement](#License-and-Acknowledgement)
### Dataset
```text
# Directory structure of the dataset
GTSRB/
  ├── train/
  │    ├── 00000/
  │    │    ├── 00000_00000.ppm
  │    │    ├── 00000_00001.ppm
  │    │    ├── ...
  │    │    └── GT-00000.csv
  │    ├── 00001/
  │    ├── ...
  │    ├── 00042/
  │    └── Readme-Images.txt
  ├── test/
  │    ├── Images/
  │    │    ├── 00000.ppm
  │    │    ├── 00001.ppm
  │    │    └── ...
  │    └── GT-final_test.csv
```


### License-and-Acknowledgement

This repository is licensed under the [Apache License 2.0](LICENSE).  

It includes code from [SwinIR](https://github.com/JingyunLiang/SwinIR), which is also licensed under the Apache License 2.0. Please also follow their licenses. 

It includes code from [LoRA](https://github.com/microsoft/LoRA#), which is licensed under the MIT License and adopts the Microsoft Open Source Code of Conduct. Please also follow their licenses. 

It contains code for training and testing models on the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_news.html).
(J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011.) 

Thanks for their awesome works.


```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 64, 64]             896
              ReLU-2           [-1, 32, 64, 64]               0
         MaxPool2d-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 64, 32, 32]          18,496
              ReLU-5           [-1, 64, 32, 32]               0
         MaxPool2d-6           [-1, 64, 16, 16]               0
            Conv2d-7          [-1, 128, 16, 16]          73,856
              ReLU-8          [-1, 128, 16, 16]               0
         MaxPool2d-9            [-1, 128, 8, 8]               0
          Flatten-10                 [-1, 8192]               0
           Linear-11                  [-1, 256]       2,097,408
             ReLU-12                  [-1, 256]               0
           Linear-13                   [-1, 43]          11,051
================================================================
Total params: 2,201,707
Trainable params: 2,201,707
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 4.00
Params size (MB): 8.40
Estimated Total Size (MB): 12.45
----------------------------------------------------------------

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           9,408
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
         MaxPool2d-4           [-1, 64, 16, 16]               0
            Conv2d-5           [-1, 64, 16, 16]          36,864
       BatchNorm2d-6           [-1, 64, 16, 16]             128
              ReLU-7           [-1, 64, 16, 16]               0
            Conv2d-8           [-1, 64, 16, 16]          36,864
       BatchNorm2d-9           [-1, 64, 16, 16]             128
             ReLU-10           [-1, 64, 16, 16]               0
       BasicBlock-11           [-1, 64, 16, 16]               0
           Conv2d-12           [-1, 64, 16, 16]          36,864
      BatchNorm2d-13           [-1, 64, 16, 16]             128
             ReLU-14           [-1, 64, 16, 16]               0
           Conv2d-15           [-1, 64, 16, 16]          36,864
      BatchNorm2d-16           [-1, 64, 16, 16]             128
             ReLU-17           [-1, 64, 16, 16]               0
       BasicBlock-18           [-1, 64, 16, 16]               0
           Conv2d-19            [-1, 128, 8, 8]          73,728
      BatchNorm2d-20            [-1, 128, 8, 8]             256
             ReLU-21            [-1, 128, 8, 8]               0
           Conv2d-22            [-1, 128, 8, 8]         147,456
      BatchNorm2d-23            [-1, 128, 8, 8]             256
           Conv2d-24            [-1, 128, 8, 8]           8,192
      BatchNorm2d-25            [-1, 128, 8, 8]             256
             ReLU-26            [-1, 128, 8, 8]               0
       BasicBlock-27            [-1, 128, 8, 8]               0
           Conv2d-28            [-1, 128, 8, 8]         147,456
      BatchNorm2d-29            [-1, 128, 8, 8]             256
             ReLU-30            [-1, 128, 8, 8]               0
           Conv2d-31            [-1, 128, 8, 8]         147,456
      BatchNorm2d-32            [-1, 128, 8, 8]             256
             ReLU-33            [-1, 128, 8, 8]               0
       BasicBlock-34            [-1, 128, 8, 8]               0
           Conv2d-35            [-1, 256, 4, 4]         294,912
      BatchNorm2d-36            [-1, 256, 4, 4]             512
             ReLU-37            [-1, 256, 4, 4]               0
           Conv2d-38            [-1, 256, 4, 4]         589,824
      BatchNorm2d-39            [-1, 256, 4, 4]             512
           Conv2d-40            [-1, 256, 4, 4]          32,768
      BatchNorm2d-41            [-1, 256, 4, 4]             512
             ReLU-42            [-1, 256, 4, 4]               0
       BasicBlock-43            [-1, 256, 4, 4]               0
           Conv2d-44            [-1, 256, 4, 4]         589,824
      BatchNorm2d-45            [-1, 256, 4, 4]             512
             ReLU-46            [-1, 256, 4, 4]               0
           Conv2d-47            [-1, 256, 4, 4]         589,824
      BatchNorm2d-48            [-1, 256, 4, 4]             512
             ReLU-49            [-1, 256, 4, 4]               0
       BasicBlock-50            [-1, 256, 4, 4]               0
           Conv2d-51            [-1, 512, 2, 2]       1,179,648
      BatchNorm2d-52            [-1, 512, 2, 2]           1,024
             ReLU-53            [-1, 512, 2, 2]               0
           Conv2d-54            [-1, 512, 2, 2]       2,359,296
      BatchNorm2d-55            [-1, 512, 2, 2]           1,024
           Conv2d-56            [-1, 512, 2, 2]         131,072
      BatchNorm2d-57            [-1, 512, 2, 2]           1,024
             ReLU-58            [-1, 512, 2, 2]               0
       BasicBlock-59            [-1, 512, 2, 2]               0
           Conv2d-60            [-1, 512, 2, 2]       2,359,296
      BatchNorm2d-61            [-1, 512, 2, 2]           1,024
             ReLU-62            [-1, 512, 2, 2]               0
           Conv2d-63            [-1, 512, 2, 2]       2,359,296
      BatchNorm2d-64            [-1, 512, 2, 2]           1,024
             ReLU-65            [-1, 512, 2, 2]               0
       BasicBlock-66            [-1, 512, 2, 2]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                   [-1, 43]          22,059
================================================================
Total params: 11,198,571
Trainable params: 11,198,571
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 5.13
Params size (MB): 42.72
Estimated Total Size (MB): 47.90
----------------------------------------------------------------
```


import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('TkAgg')
from Python_code_for_GTSRB.readTrafficSigns import readTrafficSigns

trainImages, trainLabels = readTrafficSigns('./GTSRB/train/Final_Training/Images/')
print(len(trainLabels), len(trainImages))
# print(type(trainImages[41]), getattr(trainImages[42], 'shape', None))
plt.imshow(trainImages[3000])
plt.show()
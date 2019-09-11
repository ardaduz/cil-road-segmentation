import cv2
import os
import numpy as np

# X_dir = '/home/ardaduz/ETH/CIL/project/cil-road-segmentation/competition-data/training/images/'
y_dir = '/home/ardaduz/ETH/CIL/project/cil-road-segmentation/competition-data/training/groundtruth-thresholded/'

# X = sorted(os.listdir(X_dir))

y = sorted(os.listdir(y_dir))

total_sum_road = 0
total_sum_bg = 0
for file in y:
    filename = os.path.join(y_dir, file)
    label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # label_thresholded = cv2.threshold(label, 127.5, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite(filename, label_thresholded)

    is_road = label == 255
    is_bg = label == 0

    sum_road = np.sum(is_road)
    sum_bg = np.sum(is_bg)

    total_sum_road += sum_road
    total_sum_bg += sum_bg

total = total_sum_road + total_sum_bg
print(total_sum_road)
print(total_sum_road/total)
print()
print(total_sum_bg)
print(total_sum_bg/total)
print()
print(total)

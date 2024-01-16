from cmath import inf
import numpy as np
import cv2
from numba import njit
from PIL import Image
import os
import time as time

TRAIN_OK_PATH = "C:/Users/seyyi/Desktop/ok_test"
TRAIN_NOK_PATH = "C:/Users/seyyi/Desktop/nok_test"


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


train_ok_dataset = load_images_from_folder(os.path.abspath(TRAIN_OK_PATH))
train_nok_dataset = load_images_from_folder(os.path.abspath(TRAIN_NOK_PATH))

temp = np.zeros((32, 32))

for img in train_ok_dataset:
    temp = np.add(temp, img)

temp = np.divide(temp, len(train_ok_dataset))
temp = np.round(temp, 0)
temp = np.asarray(temp).astype(np.uint8)
print(temp)
Image.fromarray(temp).save(os.path.abspath(TRAIN_OK_PATH) + "/ok_average.jpg")

temp = np.zeros((32, 32))

for img in train_nok_dataset:
    temp = np.add(temp, img)

temp = np.divide(temp, len(train_nok_dataset))
temp = np.round(temp, 0)
temp = np.asarray(temp).astype(np.uint8)
Image.fromarray(temp).save(os.path.abspath(TRAIN_NOK_PATH) + "/nok_average.jpg")

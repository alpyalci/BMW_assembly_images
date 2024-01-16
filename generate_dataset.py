cfrom cmath import inf
import numpy as np
import cv2
from numba import njit, jit, prange
from PIL import Image
import os
import time as time

t = time.time()

# train and test dataset path
TRAIN_DATA_PATH = "data/train/"
TRAIN_SCREW_PATH = "data/train/screw/"
TRAIN_CABLE_PATH = "data/train/cable/"

# used C uint as standard dtype [0; 255]
DTYPE = np.uint8


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def save_screw_to_folder(images, n):
    for i, img in enumerate(images, start=n):
        Image.fromarray(img).save(TRAIN_SCREW_PATH + "screw_" + str(i) + ".jpg")


def save_cable_to_folder(images, n):
    for i, img in enumerate(images, start=n):
        Image.fromarray(img).save(TRAIN_CABLE_PATH + "cable_" + str(i) + ".jpg")


@njit(fastmath=True)
def calculate_min_mse_img(
    dataset_img, frame_thresh, dataset_threshold, start, finish, HEIGHT, WIDTH
):
    min_mse, X, Y = inf, 0, 0

    frames_thresh = []

    for x in range(start, finish - WIDTH):
        for y in range(start, finish - HEIGHT):
            frames_thresh.append((frame_thresh[x : x + WIDTH, y : y + HEIGHT], x, y))

    for frame in frames_thresh:
        for img in dataset_threshold:
            # MSE from Frame and Img
            mse = np.square(np.subtract(frame[0], img)).mean()
            if mse < min_mse:
                min_mse, X, Y = mse, frame[1], frame[2]

    if min_mse < 0.1:
        return (
            dataset_img[X : X + WIDTH, Y : Y + HEIGHT],
            frame_thresh[X : X + WIDTH, Y : Y + HEIGHT],
        )
    return np.zeros((32, 32), DTYPE), np.zeros((32, 32), DTYPE)


def create_dataset_screw(dataset):
    IMAGE_SIZE = 150
    SCREW_HEIGHT = 32
    SCREW_WIDTH = 32

    screw_dataset = load_images_from_folder(TRAIN_SCREW_PATH)
    n = len(screw_dataset) + 1
    screw_dataset_threshold = []
    screw_img_saving = []

    for data in screw_dataset:
        _, img = cv2.threshold(data, 128, 255, cv2.THRESH_BINARY)
        screw_dataset_threshold.append(img)

    for dataset_img in dataset:
        _, frame_thresh = cv2.threshold(dataset_img, 128, 255, cv2.THRESH_BINARY)

        min_mse_img, screw_threshold = calculate_min_mse_img(
            dataset_img,
            frame_thresh,
            np.asarray(screw_dataset_threshold),
            IMAGE_SIZE * 0.2,
            IMAGE_SIZE * 0.8,
            SCREW_HEIGHT,
            SCREW_WIDTH,
        )

        if np.any(min_mse_img):
            screw_img_saving.append(min_mse_img)
            screw_dataset_threshold.append(screw_threshold)

    screw_dataset.extend(screw_img_saving)
    save_screw_to_folder(screw_img_saving, n)


def create_dataset_cable(dataset):
    IMAGE_SIZE = 150
    CABLE_HEIGHT = 24
    CABLE_WIDTH = 24

    cable_dataset = load_images_from_folder(TRAIN_CABLE_PATH)
    n = len(cable_dataset) + 1
    cable_dataset_threshold = []
    cable_img_saving = []

    for data in cable_dataset:
        _, img = cv2.threshold(data, 128, 255, cv2.THRESH_BINARY)
        cable_dataset_threshold.append(img)

    for dataset_img in dataset:
        _, frame_thresh = cv2.threshold(dataset_img, 128, 255, cv2.THRESH_BINARY)

        min_mse_img, cable_threshold = calculate_min_mse_img(
            dataset_img,
            frame_thresh,
            np.asarray(cable_dataset_threshold),
            IMAGE_SIZE * 0.3,
            IMAGE_SIZE * 0.7,
            CABLE_HEIGHT,
            CABLE_WIDTH,
        )

        if np.any(min_mse_img):
            cable_img_saving.append(min_mse_img)
            cable_dataset_threshold.append(cable_threshold)

    cable_dataset.extend(cable_img_saving)
    save_cable_to_folder(cable_img_saving, n)


train_dataset = load_images_from_folder(TRAIN_DATA_PATH)
# create_dataset_screw(train_dataset)
create_dataset_cable(train_dataset)
print(time.time() - t)

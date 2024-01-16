import numpy as np
import cv2
from numba import njit
from PIL import Image
from os import path, listdir
import time as time

t = time.time()

# train and test dataset path
TRAIN_OK_PATH = "data/train/OK/ok_average.jpg"
TRAIN_NOK_PATH = "data/train/NOK/nok_average.jpg"
TEST_DATA_PATH = "data/test/"

OK_RESULT_PATH = "data/result/OK/"
NOK_RESULT_PATH = "data/result/NOK/"


def load_avg_image_from_folder(filepath):
    img = cv2.imread(path.join(filepath), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img


def load_images_from_folder(folder):
    imgRead = cv2.imread
    joinPathOs = path.join
    grayscale = cv2.IMREAD_GRAYSCALE
    images = []
    for filename in listdir(folder):
        img = imgRead(joinPathOs(folder, filename), grayscale)
        if img is not None:
            images.append(img)
    return np.asarray(images)


def save_results_to_folder(ok_images, nok_images):
    ok_filepath = OK_RESULT_PATH + "ok_"
    nok_filepath = NOK_RESULT_PATH + "nok_"
    fromArray = Image.fromarray

    for i, img in enumerate(ok_images):
        fromArray(img).save("".join([ok_filepath, str(i), ".jpg"]))

    for i, img in enumerate(nok_images):
        fromArray(img).save("".join([nok_filepath, str(i), ".jpg"]))


@njit(
    "UniTuple(List(u1[:,:]), 2)(u1[:,:,:], u1[:,:,:], u1[:,:], u1[:,:], u1, u1)",
    fastmath=True,
)
def calculate_min_mse_img(
    dataset,
    frame_thresholds,
    dataset_ok_threshold,
    dataset_nok_threshold,
    start,
    finish,
):
    classified_ok = []
    classified_nok = []
    IMG_SIZES = dataset_ok_threshold.shape[0]

    for i, frame_thresh in enumerate(frame_thresholds):
        # Maximum 1, because RMSE is element in [0, 1]
        ok_min_rmse = 1
        nok_min_rmse = 1

        frames_thresh = [
            frame_thresh[x : x + IMG_SIZES, y : y + IMG_SIZES]
            for x in range(start, finish)
            for y in range(start, finish)
        ]

        for frame in frames_thresh:
            rmse = np.square(np.subtract(frame, dataset_ok_threshold)).mean()
            if rmse < ok_min_rmse:
                ok_min_rmse = rmse

            rmse = np.square(np.subtract(frame, dataset_nok_threshold)).mean()
            if rmse < nok_min_rmse:
                nok_min_rmse = rmse

        # Classification: OK = 1, NOK = -1
        if ok_min_rmse < nok_min_rmse:
            classified_ok.append(dataset[i])
        else:
            classified_nok.append(dataset[i])

    return classified_ok, classified_nok


def classify_dataset(dataset):
    ADAP_THRESH = cv2.adaptiveThreshold
    GAUSSIAN = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    BINARY = cv2.THRESH_BINARY

    train_ok = load_avg_image_from_folder(TRAIN_OK_PATH)
    train_nok = load_avg_image_from_folder(TRAIN_NOK_PATH)

    ok_treshold = ADAP_THRESH(train_ok, 255, GAUSSIAN, BINARY, 15, 2)
    nok_treshold = ADAP_THRESH(train_nok, 255, GAUSSIAN, BINARY, 15, 2)

    frame_thresholds = np.asarray(
        [ADAP_THRESH(img, 255, GAUSSIAN, BINARY, 15, 2) for img in dataset]
    )

    IMAGE_SIZE = dataset[0].shape[0]
    CABLE_IMG_SIZES = ok_treshold.shape[0]
    START = IMAGE_SIZE * 0.325
    FINISH = IMAGE_SIZE * 0.675 - CABLE_IMG_SIZES

    test_ok_img_saving, test_nok_img_saving = calculate_min_mse_img(
        dataset, frame_thresholds, ok_treshold, nok_treshold, START, FINISH
    )

    save_results_to_folder(test_ok_img_saving, test_nok_img_saving)

    print("Classified " + str(len(test_ok_img_saving)) + " as OK")
    print("Classified " + str(len(test_nok_img_saving)) + " as NOK")


dataset = load_images_from_folder(TEST_DATA_PATH)
print("Given " + str(len(dataset)) + " elements to classify")
classify_dataset(dataset)
print("Finished in " + str(time.time() - t) + "s")

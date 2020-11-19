from __future__ import print_function
# import potrebnih biblioteka
import cv2
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pylab as plt


# Funkcionalnost implementirana u V1
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_optimized_channel(image):
    only_one_channel_image = image[:, :, 0]  # MENJAM U KANAL POGODNIJI ZA DETEKTOVANJE RBC
    return only_one_channel_image


def image_hsv(image):
    best_channel = image[:, :, 0]
    return best_channel


def image_ots(image_hsv_with_one_channel):
    ret, image_bin = cv2.threshold(image_hsv_with_one_channel, 0, 255, cv2.THRESH_OTSU)
    return image_bin


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def image_bin_optimized(image_NZM):
    ret, image_bin = cv2.threshold(image_NZM, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=10)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=10)


def dilate_p(image, x_kernel, y_kernel, n_iterations):
    kernel = np.ones((x_kernel, y_kernel))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=n_iterations)


def erode_p(image, x_kernel, y_kernel, n_iterations):
    kernel = np.ones((x_kernel, y_kernel))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=n_iterations)


# Funkcionalnost implementirana u OCR basic
def resize_region(region):
    resized = cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)
    return resized


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def convert_output(outputs):
    return np.eye(len(outputs))


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

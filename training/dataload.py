# typstifier
# training/dataload.py

import numpy as np
import cv2
import os
import pickle

img_size = 64

def get_index(filename):
    """Takes filename of the form index-i.png and returns index"""
    return int(filename.split("-")[0])

def get_image(image_path):
    """Get image and write to numpy tensor of shape (img_size, img_size, 1)"""
    # Initially we get 0 for black and 255 for white, so we flip this. 
    return np.expand_dims((255 - cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)) / 256.0, -1)

filenames = list(os.listdir("../data"))
indices = map(get_index, filenames)

total_img_num = len(filenames)
total_sym_num = max(indices) + 1

xs_all = np.zeros((total_img_num, img_size, img_size, 1))
ys_all = np.zeros((total_img_num, total_sym_num))

for i, f in enumerate(filenames):
    xs_all[i, ...] = get_image(f"../data/{f}")
    ys_all[i, get_index(f)] = 1.0

with open("numpydata.pkl", "wb") as f:
    pickle.dump((xs_all, ys_all), f)

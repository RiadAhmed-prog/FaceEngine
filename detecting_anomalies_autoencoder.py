import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Reshape

import cv2
import numpy as np
import pickle
import csv
import datetime
# import face_recognition
from deepface import DeepFace
import os
from datetime import datetime
from video_stream import WebcamVideoStream
import glob
import time
from deepface.commons import functions, realtime, distance as dst

from face_detector import YoloDetector
from PIL import Image

import torch
import copy
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker

import traceback

import warnings
warnings.filterwarnings('ignore')
# model = YoloDetector(target_size=1080,gpu=0,min_face=30)
# sort_tracker = Sort(max_age=5,
#                     min_hits=2,
#                     iou_threshold=0.5) # {plug into parser}

config_strongsort = 'configs/deep_sort.yaml'
device = select_device("")
cfg = get_config()
cfg.merge_from_file(config_strongsort)
use_cuda = device.type != 'cpu' and torch.cuda.is_available()
deepsort = build_tracker(cfg, use_cuda=use_cuda)

yolo_model = YoloDetector(target_size=1080, gpu=0, min_face=100)
device="cuda:0"
model = DeepFace.build_model("ArcFace")
person_names = glob.glob("/home/rmedu/Ahsan Imran/FaceEngine/faceengine/celebrity_faces/*")
faces = []
known_person_info = []
# print(person_names)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def bbox_rel(xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return [x_c, y_c, w, h]

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        cat = int(categories[i]) if categories is not None else 0

        id = int(identities[i]) if identities is not None else 0

        color = compute_color_for_labels(id)

        label = f'face | {id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


# Set the path to the image folder
image_folder = './test_faces/'

# Preprocess the images
image_files = os.listdir(image_folder)
images = []
for file in image_files:
    image_path = os.path.join(image_folder, file)
    image = cv2.imread(image_path)
    face_locations_m, points = yolo_model.predict(image)

    for bb in face_locations_m[0]:

        x1 = bb[0]
        y1 = bb[1]
        x2 = bb[2]
        y2 = bb[3]
        img_pred = image[y1:y2, x1:x2]

        img_representation = DeepFace.represent(img_pred, model=model, model_name='ArcFace', enforce_detection=False)


    # Preprocess the image as needed (e.g., resize, normalize)
    images.append(img_representation)

# Convert the image list to a NumPy array
X = np.array(images)

# Normalize pixel values between 0 and 1
X = X.astype('float32') / 255.0

# Define the autoencoder model
input_dim = X.shape[1:]
print("input_dim", input_dim)
encoding_dim = 32  # Number of neurons in the bottleneck layer

# Encoder
encoder = Sequential()
encoder.add(Input(shape=input_dim))
encoder.add(Dense(64, activation='relu'))
encoder.add(Dense(encoding_dim, activation='relu'))

# Decoder
decoder = Sequential()
decoder.add(Dense(64, activation='relu', input_shape=(encoding_dim,)))
decoder.add(Dense(np.prod(input_dim), activation='sigmoid'))
# decoder.add(Dense(np.prod(input_dim)))  # Fix here
decoder.add(Reshape(input_dim))  # Reshape to match input dimensions

# Combine the encoder and decoder to create the autoencoder
autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X, X, epochs=10, batch_size=2, shuffle=True)

# Reconstruct the images using the trained autoencoder
reconstructed_images = autoencoder.predict(X)
print(X.shape)
print(reconstructed_images.shape)
# Compute the mean squared error (MSE) between the original and reconstructed images
# mse = np.mean(np.power(X - reconstructed_images, 2), axis=(1, 2, 3))
mse = np.mean(np.power(X - reconstructed_images, 2))
print(mse)
# Set a threshold for the MSE scores to classify anomalies
threshold = 0.01

# # Visualize and analyze the results
# for i in range(len(image_files)):
#     score = mse[i]
#     if score > threshold:  # Anomaly detected
#         file = image_files[i]
#         print("Anomaly Detected:", file, "MSE Score:", score)
#         # Visualize or save the image for further analysis
#         image = X[i]
#         plt.imshow(image)
#         plt.title(file)
#         plt.show()

# Visualize and analyze the results
for i in range(len(image_files)):
    score = mse
    if score > threshold:  # Anomaly detected
        file = image_files[i]
        print("Anomaly Detected:", file, "MSE Score:", score)
        # Visualize or save the image for further analysis
        image = X[i]
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(file)
        plt.show()
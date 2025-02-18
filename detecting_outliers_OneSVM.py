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


import time
import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
# from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

yolo_model = YoloDetector(target_size=1080, gpu=0, min_face=30)
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




# # Set the path to the image folder
# image_folder = '/home/rmedu/Ahsan_Imran/FaceEngine/faceengine/test_outliers/'
# # Preprocess the images and extract features
# image_files = os.listdir(image_folder)
# print(image_files)
# images = []
# for file in image_files:
#     image_path = os.path.join(image_folder, file)
#     # print("image_path", image_path)
#     image = cv2.imread(image_path)
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)
#     # Preprocess the image as needed (e.g., resize, normalize)
#     # Extract features from the preprocessed image
    
#     # features = extract_features(image)
#     try:
#         face_locations_m, points = yolo_model.predict(image)

#     except Exception as e:
#         print(e)
#         continue
#     img_rep = None
#     # img_rep = DeepFace.represent(image, model=model, model_name='ArcFace', enforce_detection=False)
#     for bb in face_locations_m[0]:
#         x1 = bb[0]
#         y1 = bb[1]
#         x2 = bb[2]
#         y2 = bb[3]
#         img_pred = image[y1:y2, x1:x2]

#         img_rep = DeepFace.represent(img_pred, model=model, model_name='ArcFace', enforce_detection=False)
#     #     # cv2.imshow('Image', img_rep)
#     #     # time.sleep(1.0)

#     if img_rep is not None:
#         images.append(img_rep)
#     else:
#         print("f")
# # Convert the image features to a NumPy array
# X = np.array(images)

# # print(X.shape)


# # pca = PCA(n_components=1)
# # X = pca.fit_transform(X)

# # print(X.shape)

# # Train the One-Class SVM model
# svm = OneClassSVM()
# svm.fit(X)
# # isolation_forest = IsolationForest()
# # isolation_forest.fit(X)

# # Predict outliers
# predictions = svm.predict(X)

# # predictions = isolation_forest.predict(X)

# # #Perform dimensionality reduction using PCA
# # pca = PCA(n_components=2)
# # X = pca.fit_transform(X)

# # Set a threshold for anomaly scores
# # threshold = -5.0


# # Visualize the feature space and anomalies

# # Visualize and analyze the results
# for file, prediction in zip(image_files, predictions):
#     image_path = os.path.join(image_folder, file)
#     image = cv2.imread(image_path)
#     if prediction == -1:  # Anomaly detected
#         anomaly_score = svm.decision_function([X[image_files.index(file)]])[0]
#         # anomaly_score = isolation_forest.decision_function([X[image_files.index(file)]])[0]
#         thresholds = np.percentile(anomaly_scores[predictions == 1], 10)  # Update the threshold to the 10th percentile of normal instances
#         print(file, thresholds)
#         if anomaly_score < threshold:
#             print("Anomaly Detected: ", file, " Score: ", anomaly_score)
#             # Visualize or save the image for further analysis
            
# Set the path to the image folder
image_folder = '/home/rmedu/Ahsan_Imran/FaceEngine/faceengine/test_outliers/'
# Preprocess the images and extract features
image_files = os.listdir(image_folder)
print(image_files)
images = []
for file in image_files:
    image_path = os.path.join(image_folder, file)
    image = cv2.imread(image_path)
    # Preprocess the image as needed (e.g., resize, normalize)
    # Extract features from the preprocessed image
    try:
        face_locations_m, points = yolo_model.predict(image)
    except Exception as e:
        print(e)
        continue
    img_rep = None
    for bb in face_locations_m[0]:
        x1 = bb[0]
        y1 = bb[1]
        x2 = bb[2]
        y2 = bb[3]
        img_pred = image[y1:y2, x1:x2]
        img_rep = DeepFace.represent(img_pred, model=model, model_name='ArcFace', enforce_detection=False)
    if img_rep is not None:
        images.append(img_rep)
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# Convert the image features to a NumPy array
X = np.array(images)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the reduced features
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1])

# Train the One-Class SVM model
svm = OneClassSVM()
svm.fit(X)

# Predict outliers and calculate anomaly scores
predictions = svm.predict(X)
anomaly_scores = svm.decision_function(X)



# Visualize the reduced features
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=predictions, cmap='coolwarm')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Feature Visualization')

# Create color legend
legend_labels = ['Anomaly', 'Normal']
colors = scatter.get_cmap()(scatter.get_array())
patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, legend_labels)]

# Add legend to the plot
plt.legend(handles=patches)
plt.savefig("database anomaly", dpi = 320)
plt.show()
# Adaptive threshold calculation
thresholds = np.percentile(anomaly_scores, 20)  # Set the initial threshold to the 10th percentile
# print(thresholds)
# Visualize and analyze the results
for file, prediction, anomaly_score in zip(image_files, predictions, anomaly_scores):
    image_path = os.path.join(image_folder, file)
    image = cv2.imread(image_path)
    if prediction == -1 and anomaly_score < thresholds:
        print("Anomaly Detected: ", file, " Score: ", anomaly_score)
        # Visualize or save the image for further analysis

# Update the threshold based on the highest anomaly score among the normal instances
# thresholds = np.percentile(anomaly_scores[predictions == 1], 5)  # Update the threshold to the 5th percentile of normal instances
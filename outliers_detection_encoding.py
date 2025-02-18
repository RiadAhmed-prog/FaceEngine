import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

import numpy as np


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

person_names = glob.glob("/home/rmedu/Ahsan_Imran/FaceEngine/faceengine/test_out/*")
faces = []
known_person_info = []
# print(person_names)


name_list = []
for name in person_names:

    only_name = name.split("/")[-1]
    print("only_name",only_name)

    name_list.append(only_name)
    known_encoding_dict = {
        'name': only_name,
        'encoding':[],
        'av_loss':[],
        'files':[]
        }
    for root, dirs, files in os.walk(name):
        temp_images = []
        temp_encoding = []
        for index, file in enumerate(files):
            img = cv2.imread(str(name+'/'+file))
            
            # print(name+'/'+file)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            except:
                print("Skipping ", name+"/"+file)
            temp_images.append(img)                                                                                                                                   
            try:
                
                face_locations_m, points = yolo_model.predict(img)
                print("face_locations_m", face_locations_m[0])
                face_locations = []
                for bb in face_locations_m[0]:
                    # face_locations.append((bb[1], bb[2], bb[3], bb[0]))
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    img_pred = img[y1:y2, x1:x2]
                    
                    img_rep = DeepFace.represent(img_pred, model=model, model_name='ArcFace', enforce_detection=False)
                



                
            except:
                print("No face found", name+"/"+file)
                
    
            known_encoding_dict['encoding'].append(img_rep)
            known_encoding_dict['files'].append(file)


        faces.append(temp_images)
        known_person_info.append(known_encoding_dict)
print("encoding done successfully for "+str(len(known_person_info))+" people" )

with open('outliers_detection_person.pickle', 'wb') as handle:
    pickle.dump(known_person_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Save the encoding file
with open('outliers_detection_encodings.pickle', 'wb') as handle:
    pickle.dump(known_encoding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
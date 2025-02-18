import cv2
import numpy as np
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

# from utils_ds.parser import get_config
# from utils_ds.draw import draw_boxes
# from deep_sort import build_tracker

import traceback

import warnings
warnings.filterwarnings('ignore')
# model = YoloDetector(target_size=1080,gpu=0,min_face=30)
# sort_tracker = Sort(max_age=5,
#                     min_hits=2,
#                     iou_threshold=0.5) # {plug into parser}

# config_strongsort = 'configs/deep_sort.yaml'
# device = select_device("")
# cfg = get_config()
# cfg.merge_from_file(config_strongsort)
# use_cuda = device.type != 'cpu' and torch.cuda.is_available()
# deepsort = build_tracker(cfg, use_cuda=use_cuda)

yolo_model = YoloDetector(target_size=1080, gpu=0, min_face=90)
model = DeepFace.build_model("ArcFace")
person_names = glob.glob("challenge/*")
# /home/faceengine/FaceEngine/known_faces
faces = []
known_person_info = []
print(person_names)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)




name_list = []

for name in person_names:
    only_name = name.split("/")[-1]
    name_list.append(only_name)
    known_encoding_dict = {
        'name': only_name,
        'encodings': [],
        'av_loss': 0,
        'matching_counter': 0,
        'score': 0,
        'ID': None,
        'max_matched': 0
    }
    for root, dirs, files in os.walk(name):
        # Add the files list to  the all_files list
        # all_files.extend(files)
        temp_images = []
        temp_encodings = []
        # print(files)
        for file in files:
            # print(name+file)
            # img = cv2.imread(str(name + "/" + file))
            img = cv2.imread(os.path.join(name, file))

            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            except:
                print("Skipping ", name + "/" + file)
            # image = face_recognition.load_image_file(str(name + file))
            temp_images.append(img)
            t_ = time.time()
            try:
                face_locations_m, points = yolo_model.predict(img)
                # print("yolo time", time.time()-t_)
                face_locations = []
                for bb in face_locations_m[0]:
                    face_locations.append((bb[0], bb[1], bb[2], bb[3]))

                    print("bb", bb)
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    # print("x1", x1)
                    img_pred = img[y1:y2, x1:x2]
                    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite("hello.jpg",img_pred)
                    # cv2.imshow("img_pred", img_pred)
                    # encod = img_pred
                    try:
                        encod = DeepFace.represent(img_path=img_pred, model=model, model_name='ArcFace', enforce_detection=False)
                    except Exception as e:
                        print("Exception DeepFace:", e)
                # encod = face_recognition.face_encodings(img)[0]
            except:
                print("No face found ", name + file)
                continue
            # temp_encodings.append(encod)
            known_encoding_dict['encodings'].append(encod)
        faces.append(temp_images)

        known_person_info.append(known_encoding_dict)
import pickle

with open('known_person_info.pickle', 'wb') as handle:
    pickle.dump(known_person_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Save the encoding file
with open('encodings.pickle', 'wb') as handle:
    pickle.dump(known_encoding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
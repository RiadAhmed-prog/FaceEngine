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

yolo_model = YoloDetector(target_size=1080, gpu=0, min_face=30)
model = DeepFace.build_model("ArcFace")
person_names = glob.glob("known_faces/*")
faces = []
known_person_info = []
print(person_names)

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


with open('encodings.pickle', 'rb') as handle:
    known_encoding_dict = pickle.load(handle)

with open('known_person_info.pickle', 'rb') as handle:
    known_person_info = pickle.load(handle)

print("encoding successfully done for " + str(len(known_person_info)) + " people")

videos_path = '/home/rmedu/Ahsan Imran/FaceEngine/faceengine/video/*'

for folder in glob.glob(videos_path):
    print(folder)
    label_name = folder.split('/')[-1]
    print(label_name)
    for file in glob.glob(folder+"/"+'*.*'):
        # print("file", file)

        false_p = 0
        # true_p = 0
        # print(file)
        # cap = cv2.VideoCapture(file)
        # cap.set(cv2.CAP_PROP_FPS, 10)

        # while True:
        #     ret , frame = cap.read()

        #     if ret:

        cap = []

        # cap.append(WebcamVideoStream(src="rtsp://admin:team6009@192.168.0.103:554").start())
        # cap.append(WebcamVideoStream(src="rtsp://admin:team6009@192.168.0.102:554/cam/realmonitor?channel=1&subtype=0").start())

        cap.append(cv2.VideoCapture(file))

        # cap = []

        # # cap.append(WebcamVideoStream(src="rtsp://admin:team6009@192.168.0.103:554").start())
        # cap.append(WebcamVideoStream(src="rtsp://admin:team6009@192.168.0.102:554/cam/realmonitor?channel=1&subtype=0").start())

        time.sleep(2)


        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        flag = 1

        nr_sources = 1
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

        frame_limit = 5
        frame_counter = 0
        to_zero = False

        label = ""
        dict_list = []
        id_list = []
        id_del = []
        false_positives = {}
        false_positives["Imran"] = 0
        false_positives["calculated"] = True
        detection_time = {}
        detection_time["start_time"] = 0
        detection_time["detected"] = True

        with open('recognition_times.csv', mode='w') as recognition_file:
            recognition_writer = csv.writer(recognition_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            recognition_writer.writerow(['Recognized Name', 'Detection Time', 'False Positive'])
            while True:
                t = time.time()
                frames = []
                flag += 1
                if flag <= 2:
                    continue

                ret, f1 = cap[0].read()
                f2 = None
                # ret, f2 = cap[1].read()
                if f1 is None:
                    f1 = np.full((840, 640, 3), 255, dtype=np.uint8)
                if f2 is None:
                    f2 = np.full((840, 640, 3), 255, dtype=np.uint8)
                frame = cv2.hconcat([cv2.resize(f1, (840, 640)), cv2.resize(f2, (840, 640))])

                # Convert image from BGR to RGB
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                

                # for i in range(len(cap)):
                #     ret, f = cap[i].read()
                #     if f is None:
                #         f = np.full((840, 640, 3), 255, dtype=np.uint8)
                #     f = cv2.resize(f, (840, 640))
                #     frames.append(f)

                # frame = cv2.hconcat([frames[0], frames[1]])

                # img = frame

                # flag += 1
                # if flag <= 2:
                #     continue

                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                try:

                    face_locations_m, points = yolo_model.predict(img)
                    face_locations = []
                    face_encodings = []

                    dets_to_sort = np.empty((0, 6))
                    confs = []
                    for bb in face_locations_m[0]:
                        bbox = bbox_rel([bb[0], bb[1], bb[2], bb[3]])
                        face_locations.append(bbox)
                        confs.append(0.9)
                    outputs = deepsort.update(np.array(face_locations), confs, img)
                    # print("face_locations", face_locations)
                    # print("outputs", outputs)
                    if len(id_list) != 0:
                        print("id_list", id_list)

                    if len(outputs) > 0:

                        bbox_tlwh = []
                        bbox_xyxy = outputs[:, :4]
                        print("outputs", outputs[:,:4])
                        print("bbox_xyxy",bbox_xyxy)
                        identities = outputs[:, -1]
                        print("identities", identities)
                        if identities is not None:
                            for id in id_list:
                                if id not in identities:
                                    dict_del = next(iter(item for item in dict_list if item['ID'] == id), None)
                                
                                    if dict_del is not None:
                                        print("deleting ID: ", id)
                                        dict_list.remove(dict_list[dict_list.index(dict_del)])
                                    
                                    else:
                                        print("not deleting", id)
                            id_list = identities

                        c = 0
                        if detection_time["detected"]:
                            detection_time["start_time"] = time.time()
                            detection_time["detected"] = False
                        for bb in bbox_xyxy:

                            x1 = bb[0]
                            y1 = bb[1]
                            x2 = bb[2]
                            y2 = bb[3]

                            ID = identities[c]
                            c += 1

                            dict_res = next(iter(item for item in dict_list if item['ID'] == ID), None)
                            if dict_res is not None and dict_res["done"]:
                                dict_res["bounding_box"] = [x1, y1, x2, y2]
                                print("fixed ID ", dict_res["ID"])
                                print("max match", dict_res["max_match"])


                                # with open('recognition_times.csv', mode='a') as recognition_file:
                                #     recognition_writer = csv.writer(recognition_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                #     row = [maxMatchedItem['name'], time.time(), false_positives.get(ID, 0)]
                                #     recognition_writer.writerow(row)
                            else:

                                img_pred = img[y1:y2, x1:x2]
                                img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
                                cv2.imshow("img_pred",img_pred)
                                av_distance = []
                                img2_representation = DeepFace.represent(img_pred, model=model, model_name='ArcFace')

                                for info in known_person_info:
                                    distances = []

                                    for encod in info['encodings']:
                                        distance = dst.findCosineDistance(encod, img2_representation)

                                        distances.append(distance)

                                    dist = np.sum(distances) / len(distances)

                                    av_distance.append(np.min(dist))


                                if np.min(av_distance) < 0.85:

                                    maxMatchedItem = known_person_info[av_distance.index(np.min(av_distance))]
                                

                                    person_name = maxMatchedItem['name']

                                    if person_name != "Imran" and false_positives["calculated"]:
                                        false_positives["Imran"] = false_positives["Imran"] + 1
                                    else:
                                        false_positives["calculated"] = False

                                    print("f p f p",false_positives)
                                    dict_res = next(iter(item for item in dict_list if item['ID'] == ID), None)

                                    if dict_res == None:
                                        
                                        new_dict = {
                                            "ID": ID,
                                            "Name": "recognizing",
                                            "max_match": 1,
                                            "matching_name": [],
                                            "bounding_box": [x1, y1, x2, y2],
                                            "done": False,
                        

                                        }
                                        
                                        dict_list.append(new_dict)
                                    else:

                                        dict_res["bounding_box"] = [x1, y1, x2, y2]
                                        to_zero = True
                                        if not dict_res["done"]:

                                            if dict_res["max_match"] == 2:

                                                print("dict res", dict_res["matching_name"])
                                                max_item = max(dict_res["matching_name"], key=lambda x: x["match"])
                                                print("max item", max_item)
                                                dict_res["max_match"] = 0
                                                dict_res["Name"] = max_item["name"]
                                                dict_res["done"] = True
                                                

                                                end_time = time.time()
                                                print("recognizing time", end_time - detection_time["start_time"])
                                                

                                                recognition_writer.writerow(["Imran", end_time - detection_time["start_time"], false_positives["Imran"]])
                                                detection_time["detected"] = True
                            
                                            else:
                                                print("person name", person_name)
                                                print("dict res, matching ", dict_res["matching_name"])
                                                dict_check = next(iter(
                                                    item for item in dict_res["matching_name"] if
                                                    item["name"] == person_name),
                                                    None)
                                                if dict_check == None:
                                                    d = {
                                                        "name": person_name,
                                                        "match": 1
                                                    }
                                                    dict_res["matching_name"].append(d)
                                                else:
                                                    dict_check["match"] += 1
                                                print("dict check", dict_check)
                                                dict_res["max_match"] += 1
                                            
                                else:
                                    dict_res = next(iter(item for item in dict_list if item['ID'] == ID), None)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, 'ID' + str(ID) + ' : ' + 'unknown', (x1 + 20, y2 - 35),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                                    if dict_res is not None:
                                        dict_list.remove(dict_list[dict_list.index(dict_res)])


                                    else:
                                        pass

                                    
                    for dict in dict_list:
                        print("dict", dict)
                        x1, y1, x2, y2 = dict["bounding_box"]
                        label = 'ID' + str(dict["ID"]) + ':' + dict["Name"]
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1 + 20, y2 - 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255),
                                    1)
                    

                    else:
                        pass

                    

                
                

                except Exception as e:
                    print(e)
                    pass


                frame_r = cv2.resize(frame, (1024, 720))
                fps = int(1 / (time.time() - t))
                print("FPS:...........", fps)
                cv2.imshow('CCTV', frame_r)
                flag = 0

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            cap[0].stop()
            cap[1].stop()
            cv2.destroyAllWindows()
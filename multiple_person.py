
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from video_stream import WebcamVideoStream
import glob

person_names = glob.glob("known_faces/*")
faces = []
known_person_info=[]
print(person_names)

name_list=[]


for name in person_names:
    only_name = name.split("/")[-1]
    name_list.append(only_name)
    known_encoding_dict={
        'name':only_name,
        'encodings':[],
        'av_loss': 0,
        'matching_counter':0,
        'score':0
    }
    for root, dirs, files in os.walk(name):
        # Add the files list to  the all_files list
        # all_files.extend(files)
        temp_images = []
        temp_encodings = []
        # print(files)
        for file in files:
            # print(name+file)
            img = cv2.imread(str(name +"/"+ file))
            try:

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print("Skipping ",name +"/"+ file)
            # image = face_recognition.load_image_file(str(name + file))
            temp_images.append(img)
            try:

                encod = face_recognition.face_encodings(img)[0]
            except:
                print("No face found ", name + file)
                continue
            # temp_encodings.append(encod)
            known_encoding_dict['encodings'].append(encod)
        faces.append(temp_images)

        known_person_info.append(known_encoding_dict)

print(name_list)
print("encoding successfully done for " + str(len(known_person_info)) + " people")

cap= WebcamVideoStream(src="rtsp://admin:Admin123@192.168.1.201:554/Streaming/Channels/101/").start()
while True:
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)


    face_names = []
    count = [0, 0, 0]
    for face_encoding,faceLoc in zip(face_encodings,face_locations):
        c = 0

        for info in known_person_info:
            # for encoding in info['encodings']:

            # print(info['encodings'])
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(info['encodings'], face_encoding)
            dists = face_recognition.face_distance(info['encodings'], face_encoding)
            info['av_loss']=np.sum(dists)/len(dists)

            info['matching_counter']=matches.count(True)
            info['score']=info['av_loss']*(len(dists)-info['matching_counter'])
            print(info['name'],info['score'])

        scores=[]
        minLossItem = min(known_person_info, key=lambda x:x['av_loss'])
        # print(minLossItem['name'])
        if minLossItem['score']<0.5:
            name = minLossItem['name'].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 1, x2 * 1, y2 * 1, x1 * 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, name+str(round(minLossItem['score'],2)), (x1 + 20, y2 - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            # print(name)
            # markAttendance(name)

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 1, x2 * 1, y2 * 1, x1 * 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, "Unknown", (x1 + 20, y2 - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            # print()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1024, 720))
    cv2.imshow('Webcam', img)
    flag = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()

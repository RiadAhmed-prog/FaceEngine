import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from video_stream import WebcamVideoStream

# from PIL import ImageGrab

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap= WebcamVideoStream(src="rtsp://admin:Admin123@192.168.1.201:554/Streaming/Channels/101/").start()
# cap = cv2.VideoCapture("vid2.mp4")
flag = 1
while True:
    success, img = cap.read()

    flag +=1
    if flag <=2:
        # flag = 0
        continue
    '''
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    '''
    # imgS = cv2.resize(img, (0, 0), None, 1.0, 1.0)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # print(facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 1, x2 * 1, y2 * 1, x1 * 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0),1)
            cv2.putText(img, name, (x1 + 20, y2 - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            print(name)
            # markAttendance(name)

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 1, x2 * 1, y2 * 1, x1 * 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, "Unknown", (x1 + 20, y2 - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            # print()
    img = cv2.resize(img, (1024,720))
    cv2.imshow('Webcam', img)
    flag = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cap.stop()
cv2.destroyAllWindows()

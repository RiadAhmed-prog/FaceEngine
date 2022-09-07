import numpy as np
from video_stream import WebcamVideoStream
import cv2
import time
cap=[]

cap.append(WebcamVideoStream(src="rtsp://admin:Admin123@192.168.1.201:554/Streaming/Channels/101/").start())
cap.append(WebcamVideoStream(src="rtsp://admin:Admin123@192.168.1.205:554/Streaming/Channels/101/").start())

time.sleep(2)
t = time.time()

font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

while True:
    frames = []
    for i in range(len(cap)):
        ret, f = cap[i].read()
        if f is None:
            f = np.full((840, 640, 3), 255, dtype=np.uint8)
        f = cv2.resize(f, (840, 640))
        frames.append(f)

    frame = cv2.hconcat([frames[0], frames[1]])

    # frame = cv2.resize(frame, (1920, 1080))

    fps = int(1 / (time.time() - t))
    t = time.time()
    nimg = cv2.putText(frame, str(fps), org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("result", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap[0].stop()
cap[1].stop()
cv2.destroyAllWindows()

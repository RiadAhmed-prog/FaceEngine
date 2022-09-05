import time
import cv2
from video_stream import WebcamVideoStream

# start camera reading thread
ws = WebcamVideoStream(src="rtsp://admin:Admin123@192.168.1.206:554/Streaming/Channels/101/").start()

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
    # read frame
    ret, img = ws.read()

    # stop if camera fails to read frame
    if not ret:
        break
    img = cv2.resize(img,(1024,720))
    fps =int(1/ (time.time() - t))
    print(fps)
    # Using cv2.putText() method
    image = cv2.putText(img, str(fps), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("result",img)
    t = time.time()
    # press q to terminate displaying frame
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, stop the camera reading thread
ws.stop()
cv2.destroyAllWindows()
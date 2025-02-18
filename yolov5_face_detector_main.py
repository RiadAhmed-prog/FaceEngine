import cv2
from video_stream import WebcamVideoStream
from face_detector import YoloDetector
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

yolo_model = YoloDetector(target_size=1080, gpu=0, min_face=30)
# Create a new video capture object

video_capture = WebcamVideoStream(src="rtsp://admin:Admin123@10.100.100.229").start()
# Create a new Haar cascade classifier
# cap = []
# cap.append(cv2.VideoCapture('/home/rmedu/Ahsan Imran/FaceEngine/video/Jennifer lawrence/abc.mp4'))

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()
    # ret, frame = cap[0].read()
    if not ret:
        print("can not read cam")
        break
    # img = cv2.resize(frame, (1024, 720))

    # try:
    #
    #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # except:
    #     pass
    try:
        # Detect faces in the grayscale frame
        face_locations_m, points = yolo_model.predict(frame)
        # print("Face location_initial: ", face_locations_m)
        # print("Points", points)
        print("face_locations_m:", face_locations_m)
        face_locations = []
        for bb in face_locations_m[0]:
            face_locations.append((bb[0], bb[1], bb[2], bb[3]))
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]
            # img_pred = img[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            print("Face locations:", face_locations)
    except:
        pass

    # Draw a rectangle around the detected faces

    try:

        # Show the resulting frame
        imG = cv2.resize(frame, (1024, 720))
        cv2.imshow('Video', imG)
    except:
        pass

    # Check for user input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
# video_capture.stop()
# cap[0].release()
# cap.release()
video_capture.stop()
cv2.destroyAllWindows()
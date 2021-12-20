import cv2
import os
import time


face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
# cam = cv2.VideoCapture(0)#for camera
video = cv2.VideoCapture('./dataset/video-1636217474.mp4')

count = 540
while True:
    OK, frame = video.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 4)
    time.sleep(0.7)
    for (x, y, w, h) in faces:
        roi = cv2.resize(
                frame[y + 2: y + h - 2, x + 2: x + w - 2], (64, 64))
        cv2.imwrite('./dataset/image_face/pic_{}.jpg'.format(count), roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        count+=1

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cam.release()
cv2.destroyAllWindows()


import cv2
import time


face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
video = cv2.VideoCapture('./dataset/raw_data/dimaria/video/Ángel Di María (@angeldimariajm) • Ảnh và video trên Instagram.mkv')

count = 262
while True:
    OK, frame = video.read()
    time.sleep(0.2)
    faces = face_detector.detectMultiScale(frame, 1.2, 2)
    for (x, y, w, h) in faces:
        roi = cv2.resize(
                frame[y + 2: y + h - 2, x + 2: x + w - 2], (64, 64))
        cv2.imwrite('./dataset/train_data/pic_{}.jpg'.format(count), roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        count+=1

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


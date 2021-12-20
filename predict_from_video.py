import cv2
import numpy as np
from tensorflow.keras import models

model = models.load_model('./model/model_test_1.h5')

face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
lst_resutl = ['Ngọc Ánh', 'Trúc Linh', 'Neymar']
video = cv2.VideoCapture('./dataset/image_face/Facebook_17.mp4')

while True:
    OK, frame = video.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 4)
    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y: y + h, x: x + w], (64, 64))
        result = np.argmax(model.predict(roi.reshape((-1, 64, 64, 3))))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(frame, lst_resutl[result], (x+15, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

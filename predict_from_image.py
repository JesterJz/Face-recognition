import cv2
import numpy as np
from tensorflow.keras import models

model = models.load_model('./model/model_test_1.h5')

face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
lst_resutl = ['Ngoc Anh', 'Truc Linh', 'Neymar']
img_path = './dataset/source_image/anh/pic_2.jpg'

while True:
    
    img = cv2.imread(img_path, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(img_gray, 1.2, 2)
    for (x, y, w, h) in faces:
        roi = cv2.resize(img[y: y + h, x: x + w], (64, 64))
        result = np.argmax(model.predict(roi.reshape((-1, 64, 64, 3))))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(img, lst_resutl[result], (x+15, y-15),
                    cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)

    cv2.imshow('FRAME', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

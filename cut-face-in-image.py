import cv2
import os


face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt2.xml')
DIR_DATA = './dataset/source_image/'
DIR_FACE = './dataset/image_face/'

for whatever in os.listdir(DIR_DATA):

    whatever_path = os.path.join(DIR_DATA, whatever)
    # print(whatever_path)
    count = 0
    for filename in os.listdir(whatever_path):
        # print(img_path)
        img_path = os.path.join(whatever_path, filename)
        img = cv2.imread(img_path, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_detector.detectMultiScale(img_gray, 1.2, 2)
        if (len(faces) != ()):
            for (x, y, w, h) in faces:  # x la cot , y la dong
                img_face = cv2.resize(
                    img[y + 2: y + h - 2, x + 2: x + w - 2], (64, 64))
                cv2.imwrite(DIR_FACE + whatever + '/pic_{}.jpg'.format(count), img_face)
                count += 1
    print(whatever + ' co ' + str(count-1) + ' tam.')
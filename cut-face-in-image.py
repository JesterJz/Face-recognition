import cv2
import os

#vào một folder chứa hình của người đó và cắt.
#sau đó cho vào 1 folder và lựa chọn hình chất lượng để dùng
#làm data để train

face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
DIR_DATA = './test/oke'

count = 0
for filename in os.listdir(DIR_DATA):
    # print(img_path)
    img_path = os.path.join(DIR_DATA, filename)
    img = cv2.imread(img_path, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(img_gray, 1.2, 2)
    if (len(faces) != 0):
        for (x, y, w, h) in faces:  # x la cot , y la dong
            img_face = cv2.resize(
                img[y + 2: y + h - 2, x + 2: x + w - 2], (64, 64))
            cv2.imwrite('./dataset/train_data/peo_{}.jpg'.format(count), img_face)
            count += 1
print(' co ' + str(count-1) + ' tam.')
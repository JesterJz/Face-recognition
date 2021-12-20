import cv2
import os


face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt2.xml')


# for whatever in os.listdir(DIR_DATA):

#     whatever_path = os.path.join(DIR_DATA, whatever)
img_path = './dataset/source_image/neymar/pic_3.jpg'
count = 0

# for filename in os.listdir(DIR_DATA):

    # img_path = os.path.join(DIR_DATA, filename)
img = cv2.imread(img_path, 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

faces = face_detector.detectMultiScale(img_gray, 1.2, 2)
if (len(faces) != ()):
    print('faces')
while True:
    for (x, y, w, h) in faces:  # x la cot , y la dong
        # img_face = cv2.resize(
        #     img[y + 2: y + h - 2, x + 2: x + w - 2], (64, 64))
        # cv2.imwrite(DIR_FACE + 'neymar' + '/pic_{}.jpg'.format(count), img_face)
        # count += 1

        cv2.rectangle(img, (x, y), (x + w, y + h),  (0, 255, 0), 1)

    cv2.imshow('FRAME', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

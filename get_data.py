import cv2
import os


DIR_DATA = './dataset/source_image/'
DIR_FACE = './dataset/image_face/'
face_detector = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt2.xml')


# for whatever in os.listdir(DIR_DATA):
#     whatever_path = os.path.join(DIR_DATA, whatever)
#     cv2.imwrite(os.path.join(DIR_FACE + whatever) + '/pic_{}.jpg'.format(count),img)
for whatever in os.listdir(DIR_DATA):

    whatever_path = os.path.join(DIR_DATA, whatever)
    count = 0
    for filename in os.listdir(whatever_path):
        img_path = os.path.join(whatever_path, filename)
        # dong truoc cot sau
        img = cv2.imread(img_path, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
        
        for (x, y, w, h) in faces:  # x la cot , y la dong
            img_face = cv2.resize(
                img[y + 2: y + h - 2, x + 2: x + w - 2], (64, 64))
            cv2.imwrite(os.path.join(DIR_FACE + whatever) + '/pic_{}.jpg'.format(count), img_face)
            count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h),  (0, 255, 0), 1)
        # cv2.imshow('FRAME', img)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# cv2.destroyAllWindows()


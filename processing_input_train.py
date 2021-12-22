import numpy as np
import os
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import models

DATA = './dataset/train_data/'
# TEST_DATA = './dataset/image_face/'


def getdata(dir):

    train_data = []
    dict = []
    count = 0
    for whatever in os.listdir(dir):

        temp = [0]*len(os.listdir(dir))
        temp[count] = 1
        count += 1
        dict.append((whatever, temp))

        whatever_path = os.path.join(dir, whatever)
        lst_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path, filename)
            img = np.array(Image.open(filename_path))
            lst_filename_path.append((img, temp))

        train_data.extend(lst_filename_path)

    return train_data, dict


train_data, dict = getdata(DATA)
# np.random.shuffle(train_data)

# xtrain = np.array([x[0] for i, x in enumerate(train_data)])
# ytrain = np.array([x[1] for i, x in enumerate(train_data)])
# print(dict)

# model_train = models.Sequential([
#     layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.15),

#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.18),

#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.2),

#     layers.Flatten(),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(4, activation='softmax')
# ])

# model_train.compile(optimizer='adam',
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy'])
# #model_train.summary()

# model_train.fit(xtrain, ytrain, epochs=10)

# model_train.save('./model/model_test_2_4_person.h5')



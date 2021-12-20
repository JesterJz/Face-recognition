#import library
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras import models


model_train =  models.Sequential([
      layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'),
      layers.MaxPool2D((2,2)),
      layers.Dropout(0.15),

      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPool2D((2,2)),
      layers.Dropout(0.2),

      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPool2D((2,2)),
      layers.Dropout(0.2),

      layers.Flatten(),
      layers.Dense(1000, activation='relu'),
      layers.Dense(256, activation='relu'),
      layers.Dense(5, activation='softmax')
])

model_train.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics='accuracy')

model_train.summary()

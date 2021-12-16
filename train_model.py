#import library
import numpy as np
import os
from PIL import Image

TRAIN_DATA = 'data/ '

Xtrain = []
Ytrain = []

for whatever in os.listdir(TRAIN_DATA):
    whatever_path = os.path.join(TRAIN_DATA, whatever)
    list_filename_path = []
    for filename in os.listdir(whatever_path):
      filename_path = os.path.join(whatever_path)
      img = np.array(Image.open(filename_path))


    print(whatever_path)




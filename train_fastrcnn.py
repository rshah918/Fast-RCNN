'''
1: load dataset and bounding boxes
2: forward prop image
3: if bb IOU > threshold, calculate regression loss
    else: no regression loss, but background softmax score to 1, foreground to 0
'''

import glob
import numpy as np
from PIL import Image

def load_dataset():
    ''' 1: load images in a list of numpy arrays
        2: introduce noise to random images
        3: load bboxes in another numpy array, store in a file'''
    train = []
    for filepath in glob.iglob('Data/train/image_data/*.jpg'):
        img = Image.open(filepath)
        img = np.array(img)
        train.append(img)
        pass
def forward_propagate(image):
    '''forward pass image'''

def calculate_IOU():
    '''calculate intersection over union'''
    pass

def get_loss():
    pass

def train(model, images):
    '''1: for each image:
            get image label
        X[] contains all images
        Y[] contans all labels
        4: backpropagate through fc layers'''

load_dataset()

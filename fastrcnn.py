from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import random
import tensorflow as tf
import cv2
import numpy as np
import keras
from time import sleep

'''
1: Convnet + Selective Search
    -OpenCV selective search
        done
    -Pretrained network, maybe vgg
        done
2: ROI pool
    inputs: feature map, region proposals
    find a way to match the coordinates of RP and FM slice
    output: slice of FM that corresponds to Region Proposals
        done

3: BB regression and classification
    done
'''

def load_test_image():
    image = load_img('mug.jpg')
    #convert to numpy array
    image = img_to_array(image)
    #convert 3D to 4D
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #reformat and zero-center pixels
    image = preprocess_input(image)
    return image #shape (1,224,224,3)

def load_base_network():
    #Pretrained VGG16 without top layer and last maxpool layer
    model = VGG16(include_top=False, weights="imagenet")
    model.layers.pop()
    model = Model(model.input, model.layers[-1].output)
    return model #shape (1,14,14,512)

def ROIpool(featureMap, proposal, H, W, inputH, inputW):
    #Normalize proposal coordinates
    proposal_left_edge = proposal[0]/inputW
    proposal_top_edge = (proposal[1]+proposal[3])/inputH
    proposal_right_edge = (proposal[0] + proposal[2])/inputW
    proposal_bottom_edge = proposal[1]/inputH

    featureMapHeight = featureMap.shape[1]
    featureMapWidth = featureMap.shape[2]
    #calculate dimensions of ROI feature vector
    ROI_left_edge = int(featureMapWidth * proposal_left_edge)
    ROI_right_edge = int(featureMapWidth * proposal_right_edge)
    ROI_top_edge = int(featureMapHeight * proposal_top_edge)
    ROI_bottom_edge = int(featureMapHeight * proposal_bottom_edge)
    ROI_width = ROI_right_edge - ROI_left_edge
    ROI_height = ROI_top_edge - ROI_bottom_edge
    ROI_channels = featureMap.shape[-1]
    ROI_batch_size = featureMap.shape[0]
    #extract ROI feature vector
    ROI = featureMap[:,ROI_bottom_edge:ROI_top_edge, ROI_left_edge:ROI_right_edge,:]
    subwindow_width = int(ROI_width/W)
    subwindow_height = int(ROI_height/H)
    #MAXPOOL the ROI
    pool = np.zeros((ROI_batch_size,H,W,ROI_channels))
    for i in range(H):
        for j in range(W):
            xmin = int(j * subwindow_width)
            xmax = int((j+1) * subwindow_width)
            ymin = int(i * subwindow_height)
            ymax = int((i+1) * subwindow_height)
            #skip proposals that are too small
            if xmin==xmax or ymin==ymax:
                continue
            else:
                pool[:,i, j, :] = np.max(ROI[:, ymin:ymax, xmin:xmax, :], axis=(1,2))
    return pool

def fully_connected_layers(ROI_batch):
    '''Create the shared FC layers, as well as the classification and regression network heads'''
    ROI_flattened_length = ROI_batch[0].shape[1]*ROI_batch[0].shape[2]*ROI_batch[0].shape[3]
    #create layers
    input = keras.Input(shape=(ROI_flattened_length,))
    fc1 = keras.layers.Dense(4096, activation='relu')
    fc2 = keras.layers.Dense(4096, activation='relu')
    classification_head = keras.layers.Dense(2, activation='softmax', name="classification")(fc2(fc1(input)))
    regression_head = keras.layers.Dense(4, activation='relu', name="bounding_box_regression")(fc2(fc1(input)))
    #build model
    output = [classification_head, regression_head]
    model = keras.Model(inputs=input, outputs=output, name="FC layers")
    model.compile(optimizer='Adam',loss={"classification": 'binary_crossentropy', "bounding_box_regression": "mean_squared_error"},
    metrics={"classification": 'binary_accuracy', "bounding_box_regression": "mean_squared_error"})
    #forward pass for each ROI
    predictions = []
    for ROI in ROI_batch:
        ROI = ROI.flatten()
        ROI = np.array([ROI])
        ROI = ROI.reshape(1,ROI_flattened_length)
        prediction = model.predict(ROI)
        predictions.append(prediction)
        print(prediction)
    return predictions

def propose_regions():
    #use CV2 selective search algorithm for region proposal
    image = cv2.imread('mug.jpg')
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = (ss.process())
    return rects #(x, y, width, height)

def forward_propagate():
    image = load_test_image()
    model = load_base_network()
    featureMap = model.predict(image)
    rects = (propose_regions())
    ROI_batch = []
    for rect in rects:
        ROI = ROIpool(featureMap, rect, 7,7,1000,1000)
        ROI_batch.append(ROI)
    predictions = fully_connected_layers(ROI_batch)

    '''raw = cv2.imread('mug.jpg')
    cv2.rectangle(raw, (rect[0], rect[1]), (rect[2]+rect[0], rect[3]+rect[1]), (0,255,0), 3)
    cv2.imshow("image", raw)
    cv2.waitKey(0)'''
    return

forward_propagate()

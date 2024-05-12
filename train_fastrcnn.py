'''
1: load dataset and bounding boxes
2: forward prop image
3: if bb IOU > threshold, calculate regression loss
    else: no regression loss, but background softmax score to 1, foreground to 0
'''

import glob
import numpy as np
from PIL import Image
from fastrcnn import *
import csv

def load_dataset():
    '''
    Load training iamges
    '''
    train = []
    i = 0
    for filepath in glob.iglob('Data/train/image_data/*.jpg'):
        if i == 100:
            break;
        #preprocess each image, convert to a numpy array, load it into ram
        image = load_test_image(filepath)
        train.append(image)
        i = i + 1
    '''
    Load bounding boxes
    '''
    bboxes = []
    #open bounding box file and skip header row
    with open('Data/train/bbox_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        #variable that will store bbox coordinates (xmin,ymin,xmax,ymax)
        bboxes = []
        bboxes_of_current_image = []
        image_index = 0
        i = 0
        for row in csv_reader:
            if i == 100:
                break
            #extract bbox and image index from row
            filename = row[0]
            filename = filename[0:len(filename)-4]
            #dump all the bboxes for that image into the final array
            if((int(filename)-10001) > image_index):
                #convert strings to ints
                for box in bboxes_of_current_image:
                    for i in range(0, len(box)):
                        box[i] = int(box[i])
                bboxes.append(bboxes_of_current_image)
                #handle the edge case in which an image has no bounding box. append
                    #an empty list to keep indices consistent
                if((int(filename)-10001) - image_index) > 1:
                    indices_skipped = ((int(filename)-10001) - image_index)
                    for i in range(0,indices_skipped-1):
                        bboxes.append([])
                        #reset/increment variables
                        bboxes_of_current_image = []
                    image_index = image_index+indices_skipped
                else:
                    #reset/increment variables
                    bboxes_of_current_image = []
                    image_index = image_index+1
            #assign bounding box to that image
            bboxes_of_current_image.append(row[3:])
            i = i + 1
        #Handle edge case at loop termination:
        for box in bboxes_of_current_image:
            for i in range(0, len(box)):
                box[i] = int(box[i])
        bboxes.append(bboxes_of_current_image)
    #return images and bounding boxes
    return (train, bboxes)

def initialize_fully_connected_layers():
    '''Create the shared FC layers, as well as the classification and regression network heads'''
    ROI_flattened_length = 7*7*512
    #create layers
    input = keras.Input(shape=(ROI_flattened_length,))
    fc1 = keras.layers.Dense(10725, activation='relu')
    fc2 = keras.layers.Dense(2048, activation='relu')
    classification_head = keras.layers.Dense(1, activation='sigmoid', name="classification")(fc2(fc1(input)))
    #regression_head = keras.layers.Dense(4, activation='relu', name="bounding_box_regression")(fc2(fc1(input)))
    #build model
    output = [classification_head]
    model = keras.Model(inputs=input, outputs=output, name="FC_layers")
    model.compile(optimizer='Adam',loss={"classification": 'categorical_crossentropy'},
    metrics={"classification": 'binary_accuracy'})
    return model

def get_loss():
    pass

def forward_propagate(fully_connected_layers, image_path = 'face.jpg'):
    image = load_test_image(image_path)
    image[0] = image[0]/255
    model = load_base_network()
    featureMap = model.predict(image)
    print(featureMap)
    rects = propose_regions()
    ROI_batch = []
    for rect in rects[:5]:
        ROI = ROIpool(featureMap, rect, 7,7,1000,1000)
        ROI_batch.append(ROI)
    #forward pass for each ROI
    predictions = []
    for ROI in ROI_batch:
        ROI = ROI.flatten()
        ROI = np.array([ROI])
        ROI = ROI.reshape(1,7*7*512)
        prediction = fully_connected_layers.predict(ROI)
        predictions.append(prediction)
    #2D list that maps each region proposal in image space to its classifier and regression output
    predictions_map = list(zip(predictions, rects))
    predictions_map.sort(key=getClassifierOutput)
    #display the top 5 predictions
    display_predictions(image, predictions_map)
    return predictions_map

def display_predictions(image, predictions_map):
    for i in range(5):
        top_prediction = predictions_map[i][1]
        cv2.rectangle(image[0], (top_prediction[0], top_prediction[1]), (top_prediction[2]+top_prediction[0], top_prediction[3]+top_prediction[1]), (0,255,0), 3)
        cv2.imshow("image", image[0])
        cv2.waitKey(0)

def train(images, bboxes):
    #training sets
    trainX = []
    trainClassY = []
    trainRegressionY = []
    #add positive samples to training sets
    base_network = load_base_network()
    for (image, bbox) in zip(images, bboxes):
        image[0] = image[0]/255
        featureMap = base_network.predict(image)
        for box in bbox:
            try:
                ROI = ROIpool(featureMap, box, 7,7,612,408)
                trainX.append(ROI.flatten())
                trainClassY.append(1.0)
                trainRegressionY.append([0.0, 0.0, 0.0, 0.0])
                #negative samples
                ROI = ROIpool(featureMap, [0, 0, 100, 100], 7,7,612,408)
                trainX.append(ROI.flatten())
                trainClassY.append(0.0)
                trainRegressionY.append([0.0, 0.0, 0.0, 0.0])
            except:
                pass
    #initialize fully connected layers
    model = initialize_fully_connected_layers()
    #model.load_weights('FaceDetectorWeights.h5')

    #compile and train
    model.compile(optimizer='Adam',loss={"classification": 'binary_crossentropy'},
    metrics={"classification": 'binary_accuracy'})
    history = model.fit(np.array(trainX), {'classification': np.array(trainClassY)}, epochs=80)
    #save model
    model.save_weights('fastrcnn.h5')

def inference():
    image_path = 'Data/train/image_data/10037.jpg'
    model = initialize_fully_connected_layers()
    model.load_weights("fastrcnn.h5")
    predictions_map = forward_propagate(model, image_path)
    print("Prediction Map: ", predictions_map)


#images, bboxes = load_dataset()
#train(images, bboxes)
inference()

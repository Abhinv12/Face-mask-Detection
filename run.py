from tensorflow.keras.models import load_model
import cv2
import numpy as np
import detect_face
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Face mask detection')
parser.add_argument('--image', type=str, help = 'Path of image', default = 'image.jpg')
args = parser.parse_args()

def predict(image, coords):
    print('Loading model')
    model = load_model('model.h5')
    predictions = []
    for (x,y,w,h) in coords:
        img = image[y:y+h,x:x+w]
        img = cv2.resize(img, (28,28))
        img = np.expand_dims(img, axis = 0)
        pred = model.predict_classes(img)
        predictions.append(pred)
    return predictions

def draw(image, coords, preds):
    for i in range(len(coords)):
        x,y,w,h = coords[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if preds[i][0] == 1:
            text = 'no mask'
        else:
            text = 'mask'
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x,y)
        fontScale = 1
        fontColor = (0,0,255)
        lineType = 2

        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
    return image
        

image = cv2.imread(args.image)
coords = detect_face.detect(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

preds = predict(image, coords)
image = draw(image, coords, preds)
cv2.imshow('',cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#window_name='image'
#cv2.imshow(window_name, image) 
cv2.waitKey(0)

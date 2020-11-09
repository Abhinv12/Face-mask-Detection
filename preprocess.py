import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

dirs_name = df['class'].unique()

for i in range(len(df)):
    sample = df.iloc[i]
    image = cv2.imread('dataset/images/'+sample['filename'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not os.path.exists('faces/'+sample['class']):
        os.mkdir('faces/'+sample['class'])
    x1,x2,y1,y2 = sample['xmin'], sample['xmax'], sample['ymin'], sample['ymax']
    face = image[y1:y2, x1:x2]
    if face.shape[0] > 28 or face.shape[1] > 28:
        plt.imsave('faces/'+sample['class'] + '/'+sample['class']+'_'+str(i)+'.jpg', face)
        print(sample['filename'])

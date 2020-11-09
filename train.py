import cv2
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale = 1./255)

train_data = train_gen.flow_from_directory('faces',class_mode = 'categorical', classes=['with_mask','without_mask'], target_size = (28,28), batch_size = 64, shuffle = True)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3), padding = 'same'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation = 'softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit_generator(train_data, epochs = 10, steps_per_epoch = 1634//64)
model.save('model.h5')

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import keras.backend as K
import cv2
import os
from cv2 import imread, imwrite
import math

X = []
dim = (770,770)
for subdir, dirs, filess in os.walk("C:\\Users\\sachg\\Pictures\\transformed"):
    for image in filess:
        array = cv2.imread(os.path.join(subdir, image), 1)
        z_array = cv2.resize(array, dim, interpolation = cv2.INTER_AREA)
        z_array = z_array.astype("float32")/255.0
        X.append(z_array)

    
X= np.array(X)
X_train = X[:306, :, :, :]


X_test = X[306:, :, :, :]




oldY = [2,1,1,2,2,1,2,2,1,2,2,2,1,1,1,1,2,1,2,1,2,1,1,2,2,1,2,2,2,2,1,1,2,1,1,2,2,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,2,1,2,1,2,2,2,1,1,1,2,2,2,1,2]
Y= []
for i in oldY:
    Y.extend([i,i,i,i,i,i])
Y_train = np.array(Y[:306])
Y_train = to_categorical(Y_train)
Y_train = Y_train[:, 1:3]

Y_test = np.array(Y[306:])
Y_test = to_categorical(Y_test)
Y_test = Y_test[:, 1:3]

model = Sequential()
model.add(Conv2D(32, (3,3), activation = "relu", input_shape = (770, 770, 3), padding= "same"))
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Conv2D(32, (3,3), activation = "relu", padding = "same"))
model.add(MaxPool2D(pool_size= (5,5)))
model.add(Conv2D(32, (3,3), activation = "relu", padding = "same"))
model.add(MaxPool2D(pool_size= (7,7)))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(2, activation = "sigmoid"))
model.compile(loss = "binary_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])
#print(model.summary())
       
model.fit(X_train, Y_train,
          batch_size = 4,
          epochs = 5,
          validation_data = (X_test, Y_test),
          shuffle=True)

model.evaluate(X_test, Y_test)







print("Done")

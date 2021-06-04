#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf

path = r"C:\\Users\\mrtkr\\Desktop\\Mask Detection\\dataset"

classList = os.listdir(path)
numberOfClasses = len(classList)
imageSize = 32

#%%
images = []
classes = []

for i in classList:
    imageList = os.listdir(path + "\\" + i)
    for j in imageList:
        img = cv2.imread(path + "\\" + i + "\\" + j)
        print(j)            
        img = cv2.resize(img, (32, 32))
        images.append(img)
        classes.append(i)
        
# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, classes, test_size = 0.5, random_state = 0)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

#%%
def preProcessImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    
    return img

#%%
x_train = np.array(list(map(preProcessImage, x_train)))
x_test = np.array(list(map(preProcessImage, x_test)))
x_validation = np.array(list(map(preProcessImage, x_validation)))

x_train = x_train.reshape(-1, 32, 32, 1)
x_test = x_test.reshape(-1, 32, 32, 1)
x_validation = x_validation.reshape(-1, 32, 32, 1)

#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)
y_train = le.fit_transform(y_train)
y_validation = le.fit_transform(y_validation)

y_test = to_categorical(y_test, 131)
y_train = to_categorical(y_train, 131)
y_validation = to_categorical(y_validation, 131)

#%%
dataGen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            rotation_range=10)
dataGen.fit(x_train)

#%%
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(input_shape = (32,32,1), filters = 20, kernel_size = (5,5), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 40, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=80, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=1024, activation = "relu"))
model.add(Dense(units=512, activation = "relu"))
model.add(Dense(units=131, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
"""hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=250),
                          validation_data=(x_validation, y_validation),
                          epochs = 25)"""
model.fit(x_train, y_train, epochs = 15)

#%%
model.save("model.model")
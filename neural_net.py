import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
path = "data\\English\\Fnt\\digit_font\\"
images = glob.glob(f"{path}*\*.png",)
list_images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in images]

labels_images = []
for i in glob.glob(f"{path}*"):
    m = re.search(r'(?<=Sample)\d+',str(i))
    count  = [int(m.group(0))-1]*len(glob.glob(f"{i}\*.png"))
    labels_images += count
labels_images = np.array(labels_images)

list_images = [cv2.bitwise_not(cv2.resize(i,(28,28))) for i in list_images] 
list_images = np.array(list_images)

y = pd.get_dummies(labels_images)
y = np.array(y)

def imshow(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

X_train, X_test,y_train, y_test = train_test_split(list_images, labels_images)
#Normalize the train
scale = MinMaxScaler()
scale.fit(X_train.reshape((7620,-1)))
X_train = scale.transform(X_train.reshape((7620,-1)))
X_test = scale.transform(X_test.reshape((2540,-1)))
#reshape data to fit model
X_train = X_train.reshape(7620,28,28,1)
X_test = X_test.reshape(2540,28,28,1)
#one-hot encode target column
y_train = np.array(pd.get_dummies(y_train, dtype = np.float64))
y_test = np.array(pd.get_dummies(y_test, dtype = np.float64))


model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2),strides = (1,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (1,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, epochs=3,validation_data = (X_test, y_test))

model.save('EnglishFnt.h5')
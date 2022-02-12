import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

data = np.load("./trainingDataTarget/data.npy")
target = np.load("./trainingDataTarget/target.npy")

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

print(train_data.shape)

# Convolutional Neural Network Model (Deep Learning)
noOfFilters = 64
sizeOfFilter1 = (3,3)
sizeOfFilter2 = (3,3)
sizeOfPool = (2,2)
noOfNode = 64

model = Sequential()
model.add((Conv2D(32, sizeOfFilter1, input_shape=data.shape[1:], activation="relu")))
model.add((Conv2D(32, sizeOfFilter1, activation="relu")))
model.add(MaxPooling2D(pool_size=sizeOfPool))

model.add((Conv2D(64, sizeOfFilter2, activation="relu")))
model.add((Conv2D(64, sizeOfFilter2, activation="relu")))
model.add(MaxPooling2D(pool_size=sizeOfPool))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(noOfNode, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuray"])

print(model.summary())








# https://www.youtube.com/watch?v=x-bPmiC23E8
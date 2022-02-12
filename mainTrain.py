import cv2
import os
import numpy as np
from keras.utils import np_utils

datapath = "Dataset"

classes = os.listdir(datapath)

labels = [i for i in range(len(classes))]

label_dict = dict(zip(classes,labels))

# Load the Data
img_size = 32
data  = []
target = []

facedata = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + facedata)

for category in classes:
    folder_path = os.path.join(datapath,category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        faces = cascade.detectMultiScale(img)
        try:
            for x, y, w, h in faces:
                sub_face = img[y: y+h, x: x+w]
                grey = cv2.cvtColor(sub_face,cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(grey,(img_size,img_size))
                data.append(resized)
                target.append(label_dict[category])
        except Exception as e:
            print("Exception: ", e)

# Processing
data = np.array(data)/255.0

# Reshape
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))

target = np.array(target)

new_target = np_utils.to_categorical(target)

np.save("./trainingDataTarget/data", data)
np.save("./trainingDataTarget/target", new_target)

print("done")

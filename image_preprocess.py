import cv2
import numpy as np
import pickle
import os

path = os.getcwd()+"\\pics\\"

images = []

print("Preprocessing...")
for file in os.listdir(path):

    img = cv2.imread(path+file)
    img = cv2.resize(img,(64,64))
    images.append(img)

X_original = np.array(images)

with open("images.pickle","wb") as f:
    pickle.dump(X_original,f)

print("Preprocessing Done!")

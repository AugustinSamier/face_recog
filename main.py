import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from math import sqrt
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

from processing import *
from KNN import *
from CNN import *


def main():
    X_train,y_train=extract_data("DataFaces")
    X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2,random_state=1,stratify=y_train)
    X_train_faces=[extract_face(img) for img in X_train]
    X_test_faces=[extract_face(img) for img in X_test]
    labels={label:i for i,label in enumerate(sorted(set(y_train)))} #list(set(y_train)) sort les val uniques en list et sorted les range dans l'ordre alphabétique 
    model=CNN_train(X_train=X_train_faces,y_train=y_train,nb_epoch=150,labels=labels,batch_size=32,patience=40,val_split=True)
    n=len(os.listdir("Models"))
    torch.save(model.state_dict(), f"model-{n}.pth")
    print(f"Sauvegardé sous le nom : model-{n}.path")



if __name__=="__name__":
    main()
import os
import cv2
import numpy as np

from CNN import *
from KNN import *

def extract_data(path,create_test=None):
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]

    file=os.listdir(path)

    for entity in file:
        try:
            entityPath=os.listdir(f"{path}/{entity}")
            for i,data in enumerate(entityPath):
                try:
                    current_img=cv2.imread(f"{path}/{entity}/{data}")
                    if current_img is not None:
                        if (create_test is not None and ((len(entityPath)>1) and (i==0))):
                            X_test.append(current_img)
                            y_test.append(entity)
                        else:
                            X_train.append(current_img)
                            y_train.append(entity)
                    else:
                        print(f"{path}/{entity}/{data} is None type")
                except Exception as e:
                    print(f"Erreur avec {path}/{entity}/{data} : {e}")
        except Exception as ex:
            print(f"Erreur avec {path}/{entity} : {ex}")
    
    print(f"Number of training sample : {len(X_train)}\n")
    if create_test is not None:
        print(f"Number of test sample : {len(X_test)}\n")
        return X_train,y_train,X_test,y_test
    else:
        return X_train,y_train



def imgProcess(img,withTorch=None,tr=True):
    if not isinstance(img,np.ndarray):
        print("convertion de l'img")
        img=cv2.imread(img)
    
    #img=extract_face(img) #applique haar cascade
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #RGB -> Gray -> dim(255,255,3)=255*255*3 val de couleur = 195075 vals -> dim(255,255,1) -> 255*255*1 val de couleur = 65025 vals pour l'exemple du 255*255
    img=cv2.resize(img,(100,100)) #dim -> dim(100,100,1) -> 100*100*1 = 10000 val de couleur
    
    if withTorch is None:
        img=img.flatten() #2D -> 1D vec de (10000,1)
    else:
        if tr is not None:
            img=transform(img)
        else:
            img=transform_test(img)
        img=img.unsqueeze(0)
    return img

def imgPadding(img,padding):
    size=img.shape[0]
    if padding is not None:
        size=size+2*padding
        newImg=np.zeros((size,size,3),dtype=np.uint8)
        for x in range(padding,size-padding):
            for y in range(padding,size-padding):
                newImg[x][y]=img[x-padding][y-padding]
    else:
        newImg=img
    return newImg

def max_pooling(img,dim=2,padding=None,stride=1):
    if padding is not None:
        img=imgPadding(img,padding)
    
    newDim=((img.shape[0]-dim)//stride)+1
    newImg=np.zeros([newDim,newDim,3])
    
    for rgb in range(img.shape[2]):
        for x in range(0,(img.shape[0]//dim)*dim,stride):
            for y in range(0,(img.shape[0]//dim)*dim,stride):
                max=0
                for h in range(dim):
                    for l in range(dim):
                        if img[x+l][y+h][rgb]>max:
                            max=img[x+l][y+h][rgb]
                newImg[((x-dim)//stride)+1][((y-dim)//stride)+1][rgb]=max
    
    return newImg

def average_pooling(img,dim=2,padding=None,stride=1):
    if padding is not None:
        img=imgPadding(img,padding)
    
    newDim=((img.shape[0]-dim)//stride)+1
    newImg=np.zeros([newDim,newDim,3])
    
    for rgb in range(img.shape[2]):
        for x in range(0,(img.shape[0]//dim)*dim,stride):
            for y in range(0,(img.shape[0]//dim)*dim,stride):
                somme=0
                for h in range(dim):
                    for l in range(dim):
                        somme+=img[x+l][y+h][rgb]
                newImg[((x-dim)//stride)+1][((y-dim)//stride)+1][rgb]=somme//(dim*dim)
    
    return newImg

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(img):
    if not isinstance(img,np.ndarray):
        img=cv2.imread(img) #si on passe un path ça read l'img

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60,60)
    )

    if len(faces)==0:
        return img #retourne l'img si aucun visage n'est detecté
    
    #si je veux detecter tous les visages présents je dois faire une boucle for x,y,w,h in faces: et return une liste d'img crops
    x,y,w,h=faces[0] #faces retourne n tuples (x,y,h,w) correspondants aux coords du visage
    face=img[y:y+h,x:x+w] #retourne l'img crop aux coords detectées
    return face
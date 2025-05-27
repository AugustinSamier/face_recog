import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

from processing import *
from processing import imgProcess
from KNN import *

def CNN_train(X_train,y_train,nb_epoch,labels,batch_size,model=None,patience=5,val_split=None):
    if model is None: #créé un model si on en donne pas
        model=make_model(len(list(set(y_train))))
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #set le device to the GPU or CPU
    model.to(device) #envoie le model au device

    if val_split is not None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,random_state=1) #prend 20% du dataset en validation
        val_dataset = list(zip(X_val, [labels[y] for y in y_val])) #validation dataset
        val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=lambda x: x)
        best_val_acc = 0
        wait = 0

    y_train_tensor=torch.tensor([labels[y] for y in y_train])

    #X_train_tensor=[imgProcess(img,withTorch=True) for img in X_train] #applique l'imgprocess en mode torch aux images train
    #X_train_tensor=torch.cat(X_train_tensor,dim=0) #stock toutes les images stockées dans X_train_tensor dans un conteneur tensor = créé le batch de taille len(X_train)
    
    #dataset=TensorDataset(X_train_tensor,y_train_tensor) #assemble nos data pour créer le dataset tensor
    dataset=list(zip(X_train,y_train_tensor))
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=lambda x: x) #shuffle -> pour changer l'ordre des pairs img-label pour pas qu'on choppe des batchs uniquement du même label
    #le dataloader va séparer le dataset de taille N en (N//batch_size)+1 si N%batch_size != 0 -> il y aura N//batch_size batchs complets et un batch plus petit avec les data restantes
    #on peut rajouter le param drop_last=True si on veut supprimer le dernier batch -> +stabilité et certaines couches demandent un full batch (BatchNorm par ex)

    #+ le batch_size est élevé + ce sera précis mais + ça consommera

    criterion = torch.nn.CrossEntropyLoss() #fonction de coût/perte à minimiser -> crossentropy utilisée pour la classification multiclasse
    #applique softmax à la prédiction (qui sont des scores) pour transformer les scores en probabilités puis compare les proba à la classe cible et renvoie une erreur.
    #plus la probabilité de la bonne classe est faible, plus la perte sera grande
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #lr = learning rate -> plus c'est petit plus c'est progressif et donc plus il aura besoin d'apprentissage mais plus il sera précis
    #Adam = Adaptive Moment Estimation -> optimiseur pour réduire la perte. Adam adapte le lr localement donc pas besoin de régler le lr très finement. Utilise le SGD (Stochastic Gradient Descent)
    #SGD -> minimisation de la perte pour chaque exemple individuellement là ou le classique GD minimise la perte globale -> SGD plus bruité (moins stable) mais + rapide

    model.train() #active le mode training
    for epoch in range(nb_epoch): # 1 epoque = le moment où le modèle a vu tout le dataset donc une époque = tous les (N//batch_size)+1 batchs
        # plus il y a d'epoch + le model va apprendre et réduire l'erreur mais attention à l'overfitting
        #-> petit dataset(>1000) = 10 to 50 epochs -> moyen dataset (1000 to 100k) = 20 to 100 et grand dataset (>100k) -> 10 to 50
        #SINON -> utiliser early stopping pour être précis qui stop quand la précision sur validation baisse ou n'augmente plus -> à coder soit même
        total_loss=0.0
        correct=0
        total=0

        for batch in dataloader:
            X_raw,labels_batch=zip(*batch)
            img=[imgProcess(image,withTorch=True) for image in X_raw]
            img=torch.cat(img,dim=0)
            img=img.to(device) #envoie les img au device
            labels=torch.tensor(labels_batch).to(device) #envoie les labels au device, tout à besoin d'être sur le même device pour être traité

            outputs=model(img) #prédiction des img (forward)
            loss=criterion(outputs,labels) #calcul de la perte avec le criterion (ici crossentropy)

            #rétropropagation (backward)
            optimizer.zero_grad() #remet les gradients à 0
            loss.backward()  #calcul les nouveaux gradients

            optimizer.step() #mise à jour des poids

            #partie statistiques (facultatif mais utile)
            total_loss+=loss.item() #additionne la loss de chaque epoch
            _,predicted=torch.max(outputs,1) #prend le score max de la dim 1 de output qui correspond à la classe et prend uniquement la classe associée à ce score max = les classes prédites
            correct+=(predicted==labels).sum().item() #compte combien de ces preds sont correctes
            total+=labels.size(0) #compte le nb d'img traitées dans cette epoch (labels.size(0)=32 ici vu que c'est le batch_size)
            
        acc=100*correct/total #bonnes prédictions sur l'ensemble de l'epoch en %
        print(f"\nEpoch : {epoch+1}/{nb_epoch}\n Perte :{total_loss:.4f}\n Précision : {acc:.2f}%")
        
        if val_split is not None:
            true,false,tot=CNN_evaluate(model=model,loader=val_loader,device=device)
            val_acc=sum(true.values())/sum(tot.values()) #calcule l'accuracy du dataset validation
            if val_acc>best_val_acc:
                best_val_acc=val_acc #si l'acc de la val est meilleure on l'assigne et on reset le timer 
                wait=0
                best_model=copy.deepcopy(model.state_dict()) #on copie le meilleur model
            else:
                wait+=1
                if wait>=patience: #on attend le nombre d'epoch max sans amélioration
                    print("Early stopping activated")
                    break
            print(f"\n Val accuracy : {val_acc*100:.2f}%\n Série sans amélioration : {wait}")
            model.load_state_dict(best_model)
    
    print("Training terminé.")
    return model

def CNN_evaluate(model,X_test=None,y_test=None,labels=None,batch_size=None,loader=None,device=None):
    if device is None:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    model.eval() #active le mode evaluation

    if loader is None:
        y_test_tensor=[labels[y] for y in y_test] #remplace chaque nom dans y_test par son indice dans labels
        test_dataset=list(zip(X_test,y_test_tensor))
        loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=lambda x: x)

    """counts={} #dictionnaire pour compter les labels les plus proches et leur occurences
    for _,label in k_voisins: #dans le tableau c'est [distance,label], ici on veut que le label donc _
        counts[label]=counts.get(label,0)+1 #.get(a,b) = prend la val a si existe sinon prend la val b"""
    
    labelF=defaultdict(int) #équivalent à faire labelF={} puis plus tard labelF[label]=labelF.get(label,0)+1 #pour les fausses prédictions
    labelC=defaultdict(int) #pour les correctes
    labelT=defaultdict(int) #pour les totaux

    with torch.no_grad(): #pas de backward (juste forward = pred)
        for batch in loader:
            X_raw,labels_batch=zip(*batch)
            img=[imgProcess(image,withTorch=True,tr=None) for image in X_raw]
            img=torch.cat(img,dim=0)
            img=img.to(device)
            labels=torch.tensor(labels_batch).to(device)

            outputs=model(img)
            _,pred=torch.max(outputs,1)
            for i in range(len(labels_batch)):
                label=labels[i].item()
                pred[i]=pred[i].item()

                if pred[i]==label:
                    labelC[label]+=1
                else:
                    labelF[label]+=1
                labelT[label]+=1
            
    print(f"Total precision : {sum(labelC.values())/sum(labelT.values())*100:.2f}%")

    precisions={}
    for label in labelT:
        tot=labelT[label]
        correct=labelC.get(label,0)
        precisions[label]=correct/tot if tot>0 else 0
    prec_sorted=sorted(precisions.items(),key=lambda x: x[1])
    labels_ids,values= zip(*prec_sorted)

    id2label = {v: k for k, v in labels.items()}
    class_names=[id2label[i] for i in labels_ids]

    plt.figure(figsize=(12, 5))
    plt.bar(class_names, values, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("Précision par classe")
    plt.title("Performance du modèle par classe")
    plt.tight_layout()
    plt.show()
    return labelC,labelF,labelT

def CNN_inf(img_path,model,labels):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    img=imgProcess(img_path,withTorch=True).to(device)

    with torch.no_grad():
        output=model(img)
        _,pred=torch.max(output,1)
    
    return labels[pred.item()]


transform_test=transforms.Compose([
    transforms.ToPILImage(), #transforme en format PIL 
    transforms.ToTensor() #reconvertit l'img en format tensor
    ])

transform=transforms.Compose([
    transforms.ToPILImage(), #transforme en format PIL
    transforms.RandomHorizontalFlip(), #0.5 de proba d'inverser la gauche et la droite de l'img pour rendre le modèle invariant à la symétrie
    transforms.RandomRotation(10), #applique rota random entre -10° et +10° 
    transforms.ColorJitter(brightness=0.2,contrast=0.2), #altère aléatoirement la luminosité et le contraste de l'image
    transforms.ToTensor() #reconvertit l'img en format tensor
    ])

def make_model(nb_classes):
    model1=nn.Sequential(
        nn.Conv2d(1,16,kernel_size=3,padding=1), #(1,100,100) -> (16,100,100)           = 1*3*3*16+16 = 160
        nn.ReLU(),
        nn.MaxPool2d(2,2), #(16,100,100) -> ((100-2)//2)+1 = 49+1 = 50 donc -> (16,50,50)
        nn.Conv2d(16,32,kernel_size=3,padding=1), #(16,50,50) -> (32,50,50)             =16*3*3*32+32 = 4640
        nn.ReLU(),
        nn.MaxPool2d(2,2), #(32,50,50) -> ((50-2)//2)+1 = 25 -> (32,25,25)
        nn.Flatten(), #(32,25,25) -> (32*25*25) = (20000)
        #nn.Dropout(0.3) #pour éviter l'overfitting
        nn.Linear(20000,128),       # = 20000*128 +128 = 2560128
        nn.ReLU(),
        nn.Linear(128,nb_classes)       # = 128*nb class + nb class
    )   # 160+4640+2560128+129*nbclass = 2 564 928 + 129*nbclass PARAMS

    model2=nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),  # → (16, 100, 100)
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                         # → (16, 50, 50)

        nn.Conv2d(16, 32, kernel_size=3, padding=1), # → (32, 50, 50)
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                         # → (32, 25, 25)

        nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (64, 25, 25)
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                         # → (64, 12, 12)

        nn.AdaptiveAvgPool2d((4, 4)),               # → (64, 4, 4) (plus robuste aux tailles)
        nn.Flatten(),                               # → 64*4*4 = 1024

        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(128, nb_classes)
    )

    #modèle VGG-Face modifié
    model=nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1,100,100) → (32,100,100)
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),  # → (32,100,100)
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                          # → (32,50,50)

        # Block 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                          # → (64,25,25)

        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                          # → (128,12,12)

        # Adaptive pooling for flexibility
        nn.AdaptiveAvgPool2d((4, 4)),                # → (128,4,4)
        nn.Flatten(),                                # → 2048

        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.4), 

        nn.Linear(256, nb_classes)
    )

    return model       
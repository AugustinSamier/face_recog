from processing import *
from CNN import *

def distEuclid(img1,img2): #p1(a1,a2,a3...,a10000) p2(b1,b2,b3...,b10000) car chaque image a 10000 val de couleur
    #img1 = img1.astype(np.int32) #pour avoir des valeur négatives sinon en uint8 c'est de 0 à 255 (pas negatif)
    #img2 = img2.astype(np.int32)
    diff=img1.astype(np.int32)-img2.astype(np.int32) #opti

    #distance euclidienne entre deux points (en 2D) p1(x1,y1) et p2(x2,y2): d=hypothénuse du triangle rectangle
    #ainsi d²=a²+b² et d=sqrt(a²+b²) avec a=abs(x1-x2) et b=abs(y1-y2) DONC -> d=sqrt((x1-x2)²+(y1-y2)²)
    #en n dim : d=sqrt((a1-b1)²+(a2-b2)²...+(an-bn)²) = sqrt(sum[1;N]((a-b)²))
    #return sqrt(sum((a-b)**2 for a,b in zip(img1,img2))) #zip -> pour deux listes
    return np.sqrt(np.sum(diff**2)) #opti

def knn_predict(X_train,y_train, newImg,k):
    distances=[]
    for i in range(len(X_train)):
        dist=distEuclid(X_train[i],newImg) #on calcule la distance entre la nouvelle image et toutes les autres images
        distances.append((dist,y_train[i])) #on stock la distance avec le label de chaque image à laquelle on a calculé la distance
    
    distances.sort(key=lambda x: x[0]) #on trie la liste de distances dans l'ordre croissant comme ça les derniers termes sont les plus proche de la nouvelle image
    k_voisins=distances[:k] #on prend les k plus proches voisins

    counts={} #dictionnaire pour compter les labels les plus proches et leur occurences
    for _,label in k_voisins: #dans le tableau c'est [distance,label], ici on veut que le label donc _
        counts[label]=counts.get(label,0)+1 #.get(a,b) = prend la val a si existe sinon prend la val b

    return max(counts,key=counts.get) #compare les valeurs mais renvoie le label du dictionnaire

def knn_evaluate(processed_train,y_train,processed_test,y_test,k,n):
    knn_results=[]
    true=0
    if n is None:
        n=len(processed_test)

    print(f"Number of sample set to : {n}")
    
    for i,img in enumerate(processed_test[:n]):
        print(f"Image KNN : {i}")
        knn_results.append(knn_predict(processed_train,y_train,processed_test[i],k))
        if knn_results[i]==y_test[i]:
            true+=1
    print(f"Accuracy (%) : {100*true/n}")

    return knn_results

def KNN(X_train,y_train,X_test,y_test,k=3,n=None):
    processed_train=[]
    for img in X_train:
        processed_train.append(imgProcess(img))

    processed_test=[]
    for img in X_test:
        processed_test.append(imgProcess(img))
    print("Processing done.\n")

    knn_result=knn_evaluate(processed_train,y_train,processed_test,y_test,k,n)
    return knn_result
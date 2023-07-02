"""
Python’da iris datasetini kullanarak KNN (K En yakın komşu) algoritmasının  kodları
"""

import numpy as np
import pandas as pd
import matplotlib as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


#İris veri setini çağırıyoruz.
X, y = load_iris(return_X_y=True)

#Eğitim ve test verisi olarak ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

#Ölçeklendirilme işlemi
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

score = list()

for i in range(1,20,2):
    #KNN algortimasını çalıştırıyoruz.
    knn = KNeighborsClassifier(n_neighbors=i) #k = 3

    #Normalleştirilme işlemi
    knn.fit(X_train,y_train)

    #Tahmin
    y_pred = knn.predict(X_test)

    #Confusion Matrix
    cm = confusion_matrix(y_test,y_pred)

    """
    print()
    print("---------------------------------------------")
    print("Confusion matrix K-NN için")
    print("---------------------------------------------")
    print(cm)
    print(cm[0][0] + cm[1][1] + cm[2][2]) #Doğru sonuçlar
    """
    score.append(cm[0][0] + cm[1][1] + cm[2][2])

listk = list()    
for j in range(1,20,2):
    listk.append(str(j))

skor = dict(zip(listk, score))

print("---------------------------------------------")
print("K değerlerinin doğrulukları 20'e kadar")
print("---------------------------------------------")
for k, v in skor.items():
    print("k=",k,":",v)
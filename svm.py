import numpy as np
import pandas as pd
import matplotlib as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn import svm


#İris veri setini çağırıyoruz.
X, y = load_iris(return_X_y=True)

#Eğitim ve test verisi olarak ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

#Ölçeklendirilme işlemi
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

dvm = svm.SVC() #Support Vector Machines Classification
#svr = svm.SVR() #Support Vector Machines Regression (SVR)

linear_svc = svm.SVC(kernel='linear')
kerl = linear_svc.kernel

rbf_svc = svm.SVC(kernel='rbf')
kerr = rbf_svc.kernel

poly_svc = svm.SVC(kernel='poly')
kerp = poly_svc.kernel

sigmoid_svc = svm.SVC(kernel='sigmoid')
kersig = sigmoid_svc.kernel

#Normalleştirilme işlemi
dvm.fit(X_train,y_train)
linear_svc.fit(X_train,y_train)
rbf_svc.fit(X_train,y_train)
poly_svc.fit(X_train,y_train)
sigmoid_svc.fit(X_train,y_train)


#Not: Her biri normalleştirilmiş olamlı yoksa hata veriyor.

#Tahmin
y_pred = dvm.predict(X_test)
y_predlk = linear_svc.predict(X_test)
y_predrbfk = rbf_svc.predict(X_test)
y_predpol = poly_svc.predict(X_test)
y_predsig = sigmoid_svc.predict(X_test)


#Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
cs = list()
kernels = ["SVM default"]
kernels.append(kerl)
kernels.append(kerr)
kernels.append(kerp)
kernels.append(kersig)





print()
print("---------------------------------------------")
print("Confusion matrix SVM için")
print("---------------------------------------------")
print(cm)
print()
cs.append(cm[0][0] + cm[1][1] + cm[2][2])
 

cm = confusion_matrix(y_test,y_predlk)


print()
print("---------------------------------------------")
print(f"Confusion matrix SVM {kerl} kernel için")
print("---------------------------------------------")
print(cm)
print()

cs.append(cm[0][0] + cm[1][1] + cm[2][2])

cm = confusion_matrix(y_test,y_predrbfk)

print()
print("---------------------------------------------")
print(f"Confusion matrix SVM {kerr} kernel için")
print("---------------------------------------------")
print(cm)
print()

cs.append(cm[0][0] + cm[1][1] + cm[2][2])

cm = confusion_matrix(y_test,y_predpol)

print()
print("---------------------------------------------")
print(f"Confusion matrix SVM {kerp} kernel için")
print("---------------------------------------------")
print(cm)
print()

cs.append(cm[0][0] + cm[1][1] + cm[2][2])

cm = confusion_matrix(y_test,y_predsig)

print()
print("---------------------------------------------")
print(f"Confusion matrix SVM {kersig} kernel için")
print("---------------------------------------------")
print(cm)
print()

cs.append(cm[0][0] + cm[1][1] + cm[2][2])



sonuc = dict(zip(kernels,cs))

print("---------------------------------------------")
print("Doğru sonuçlar:")
print("---------------------------------------------")
for k, v in sonuc.items():
    print(k,"kernel :",v)
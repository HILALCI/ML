"""
KullaniciID, Cinsiyet, Yas, TahminiMaas, SatinAldiMi gibi sosyal medya -
verisinden oluşan dataseti kullanarak Naive Bayes Algoritması aracılığıyla sınıflandırma uygulaması
Bu bağımsız niteliklerle bağımlı nitelik -
(satın alma davranışının gerçekleşip gerçekleşmeyeceği) tahmin edilecektir. 
"""

import numpy as np
import pandas as pd
import matplotlib as mp
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#Veri setinin okunması işlemi için
data = pd.read_csv("sosyalmedya_nb_dataset.csv")

#Kadın ve Erkek değerler str old. kabul etmedi sayısal değer istiyordu bu yüzden 0 ve 1 değerlerini atadık.
data = data.replace("Kadın", 0)
data = data.replace("Erkek", 1)

#print(data)

#Veri setindeki bağımlı ve bağımsız değişkenleri alıyoruz.
X = data.iloc[:,1:4].values #1. indexten başlamamızın nedeni Kullanıcı ID ile işimiz olmamasından dolayı
y = data.iloc[:,4:].values 
#Scikit-learn dökümantasyonunda X olarak yazıldığı için dökümana sadık kalınmıştır.

#Veriyi eğitim ve test verisi olarak parçalıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#parametre olarak  random_state=0 eklersek hep aynı sonucu verecektir.Bu işlem R'da set.seed() methodu ile aynı işlemi görecektir.

#Not: Eğim %70 , Test %30 olacak şekilde ilk olarak parçalanmıştır. Fakat en iyi sonuç vermesi için denenecektir.

#Verilerin ölçeklendirilmesi işlemi ön işleme aşaması
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Verinin Gaussian Naive Bayes yöntemi kullanılacaktır.
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#Tahmin işlemi
y_pred = gnb.predict(X_test)

#Tahmin doğruluğunun ölçülmesi için confusion matrix bakılabilir.
cm = confusion_matrix(y_test, y_pred)

#Çıktılar
v,f = np.unique(y_test, return_counts = True)
ct = np.asarray((v,f))

print()
print("Test Değerleri")
print("******")
print(ct)
print("Test satın alanların sayısı = ", ct[1][1])
print("Test satın almayanların sayısı = ", ct[1][0])
print()

#print("\n", y_pred, "\n")

v,f = np.unique(y_pred, return_counts = True)
ctp = np.asarray((v,f))

print("Tahmin Değerleri")
print("******")
print(ct)
print("Tahminen satın alanların sayısı = ", ctp[1][1])
print("Tahminen satın almayanların sayısı = ", ctp[1][0])
print()

print("Tahminin sapması veya hatası")
print("******")
print("Satın alanların sayısındaki hata = ", abs( ct[1][1] - ctp[1][1] ))
print("Satın almayanların sayısındaki hata = ", abs( ct[1][0] - ctp[1][0] ))
print("Toplam hata = ", abs( ct[1][1] - ctp[1][1] ) + abs( ct[1][0] - ctp[1][0] ))

"""
for i in y_pred:
    if i == 1:
        print("Satın Aldı.")
    else:
        print("Satın Almadı.")
"""

"""
v,f = np.unique(data.iloc[:,4:].values, return_counts = True)
ct = np.asarray((v,f))

print("Gerçek Değerler Tamamı")
print("******")
print(ct)
print("Gerçekte satın alanların sayısı = ", ct[1][1])
print("Gerçekte satın almayanların sayısı = ", ct[1][0])
print()
print("Not: Sayı farkının fazla olmasının nedeni veriyi parçaladığımız içindir.")

"""
print()
print("---------------------------------------------")
print("Confusion matrix Gaussian Naive Bayes için")
print("---------------------------------------------")
print(cm)
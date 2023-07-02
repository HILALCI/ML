import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn import tree

#İris setinin çağırmak için
iris = load_iris()

#Test ve eğitim için verileri ayırıyoruz.
X, y = iris.data, iris.target

#Karar ağacı algoritması ile sınıflandırma işlemi
dt = tree.DecisionTreeClassifier()

#Normalleştirme işlemi
dt = dt.fit(X, y)

#Görselleştirme işlemi
tree.plot_tree(dt)
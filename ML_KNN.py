 # sklearn: ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# (1) veri seti incelemesi 
cancer = load_breast_cancer()  

df= pd.DataFrame(data = cancer.data, columns= cancer.feature_names)
df["target"] = cancer.target

# (2)makine öğrenmesi modelinin seçilmesi -KNN sınıflandırıcı


# (3)modelin train edilmesi

X = cancer.data # features
y = cancer.target #target

#train test split
X_train,X_test, y_train, y_test = train_test_split(X , y, test_size= 0.3 , random_state=42)

#olceklendirme 
scaler =StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#KNN modeli oluştur v etrain et
knn = KNeighborsClassifier( n_neighbors=3) # model oluşturma komşu parametresini oluşturma*******
knn.fit(X_train ,y_train) # fit fonksiyonu verimizi ( samples + target) kulllanarak KNN algoritmasını eğitir
 
# (4) sonuçların değerlendirilmesi : tests
y_pred= knn.predict(X_test)


accuracy = accuracy_score ( y_test, y_pred)
print("doğruluk:",accuracy)

conf_matrix = confusion_matrix( y_test, y_pred)
print("confusion_matrix:")
print(conf_matrix)

#hiperparametre ayarlaması
"""
KNN : Hyperparameter = K 
K: 1,2,3, ... N
Accuracy: ½A, ½B,...
"""
accurarcy_values =[]
k_values =[]
for k in range(1, 21):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train , y_train)
    y_pred= knn.predict(X_test)
    accurarcy = accuracy_score(y_test , y_pred)
    accurarcy_values.append(accurarcy)
    k_values.append(k)
    
plt.figure()
plt.plot( k_values, accurarcy_values, marker= "o", linestyle = "-")
plt.title("k degerine göre doğruluk")
plt.xlabel("k degeri")
plt.ylabel("dogruluk")
plt.xticks(k_values)
plt.grid(True)


# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.sort ( 5 * np.random.rand(40, 1 ), axis = 0) #features
y = np.sin(X).ravel()#target

 #plt.scatter(X, y)


# add noise
y[::5] += 1 * (0.5 - np.random.rand(8))

# plt.scatter(X,y)
T= np.linspace(0, 5, 500) [:, np.newaxis]

for i, weight in enumerate( [ "uniform" ,"distance"]):

    knn = KNeighborsRegressor(n_neighbors=5, weights= weight)
    y_pred= knn.fit(X,y).predict(T)

    plt.subplot(2,1,i+1)
    plt.scatter(X, y , color= "green",label="data")
    plt.plot(T,y_pred,color="blue",label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN regressor weight= {} ".format(weight))
    
plt.tight_layout()
plt.show()






















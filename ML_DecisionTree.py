from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,plot_tree
from sklearn.metrics import accuracy_score , confusion_matrix

import matplotlib.pyplot as plt

#veri seti inceleme 
iris = load_iris()

#
X = iris.data  # features(sepal/petal uzunluğu vs.)
y = iris.target  # target Hedef: Çiçek türü (0, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)

#Decision Tree modeli oluştur ve train et 
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth = 5, random_state=42) # criterion="entropy"
tree_clf.fit(X_train, y_train)

#Decision tree evalution test
y_pred = tree_clf.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("iris veri seti ile eğitilen DT modeli doğruluğu:", accuracy)


conf_matrix = confusion_matrix(y_test, y_pred)
print("conf_matrix:")
print(conf_matrix)

plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled= True, feature_names= iris.feature_names, class_names= list(iris.target_names))
plt.show()

feature_importances = tree_clf.feature_importances_
#görselleştirme
feature_names= iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse=True)
for importance, feature_name  in feature_importances_sorted:
    print(f"{feature_name}: {importance}")
    
    
    # %%
    
    
    
    
    
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


plt.close('all')  # Tüm önceki figürleri kapat
plt.figure()      # Yeni bir figür başlat


#veri seti inceleme 
iris = load_iris()

n_clases=len(iris.target_names)
plot_colors ="ryb"

for pairidx, pair in enumerate ([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    
    X= iris.data[:, pair]
    y= iris.target
     
    clf= DecisionTreeClassifier().fit(X,y)
    
    ax= plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad = 0.5, w_pad = 0.5, pad= 2.5)
    DecisionBoundaryDisplay.from_estimator(clf, X,
                                           cmap= plt.cm.RdYlBu, 
                                           response_method= "predict", 
                                           ax=ax, 
                                           xlabel= iris.feature_names[pair[0]], 
                                           ylabel=iris.feature_names[pair[1]])
    
    for i, color in zip(range(n_clases),plot_colors):
        idx=np.where(y==i)
        plt.scatter(X [idx, 0], X [idx,1], c=color, label= iris.target_names[i],
                    cmap = plt.cm.RdYlBu, edgecolors="black"
                    )
    
    
    plt.legend()
    
    
    
    
    #%%
    
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error   # test ve pred değerleri arasındaki farkların bir şeyini hesaplıyor. Model tahminlerinin doğruluğunu ölçmek için hata hesaplar.
import numpy as np  # Matematiksel işlemler için.


diabetes = load_diabetes()     
    
X =diabetes.data #features  Bağımsız değişkenler (features)
y =diabetes.target #features  Bağımlı değişken (target, tahmin edilmek istenen değer)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state= 42)  # verileri ikiye ayırdık train ve test diye   
    
    
    
#karar ağacı regresyon modeli

tree_reg = DecisionTreeRegressor(random_state=42    )
tree_reg.fit(X_train, y_train) #eğitimi gerçekleştiriyoruz   
    
    
y_pred= tree_reg.predict(X_test)    


mse =mean_squared_error(y_test,y_pred)    
print("mse:",mse)    

rmse =np.sqrt(mse)
print("rmse:",rmse)
    
    
    
  #%%
    
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt


#create a data     
X = np.sort(5 * np.random.rand(80,1),axis=0)
y =np.sin(X).ravel()
y[::5] += 0.5 *(0.5 - np.random.rand(16)) # gürültü ekledik beş aralıkta bir çözmesi daha zor olsun diye

# plt.scatter(X,y)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X,y)    
regr_2.fit(X,y)    
    
X_test =np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1= regr_1.predict(X_test)     
y_pred_2= regr_2.predict(X_test)    
    
    
plt.figure()   
plt.scatter(X,y ,c="red", label= "blue")    
plt.plot(X_test, y_pred_1 ,color="blue", label= "max depth: 2", linewidth=2)    
plt.plot(X_test, y_pred_2 ,color="green", label= "max depth: 5", linewidth=2)        
plt.xlabel ("data")   
plt.ylabel("target")    
plt.legend()    
    
    
    
    
    
    
    
    
    
    
    
    
    
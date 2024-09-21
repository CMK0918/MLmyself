from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 載入數據集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 觀察資料
# print(pd.Series(iris))
# print(X)
# print(y)

# 區分訓練集與測試集   訓練集: 70%  測試集: 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 建模參數 
# n_neighbors: K
# weight: "uniform"/"distance"/其他
# algorithm: "auto"/"brute"/"kd_tree"/"ball_tree" 
# p: 1 曼哈頓距離/2 歐基里德距離/ 其他:明氏距離
clf = KNeighborsClassifier(n_neighbors=5, p=2, weights="distance", algorithm="brute")
clf.fit(X_train, y_train)

# 預測
y_pred = clf.predict(X_test)

print(y_test)
print(y_pred)

# 準確程度評估
print(clf.score(X_test, y_test))

# 尋找合適的K 比較好的K值期望落在樣本數的平方根裡面

# print(len(X_train)) # 105 個樣本

accuracy = []

for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k, p=2, weights="distance", algorithm="brute")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))

k_range = range(1, 100)
plt.plot(k_range, accuracy)
plt.show()
from sklearn import preprocessing  # 用於數據預處理，例如標準化
import numpy as np  # 用於數據處理
from sklearn.model_selection import train_test_split  # 用於將數據分割為訓練集和測試集
from sklearn.datasets import make_classification  # 用於生成分類數據
from sklearn.svm import SVC  # 用於支持向量機分類
import matplotlib.pyplot as plt  # 用於數據可視化

# 創建一個包含三行三列的 NumPy 陣列
a = np.array([[10, 2.7, 3.6],
              [-100, 5, -2],
              [120, 20, 40]], dtype=np.float64)

# 打印原始數據
print(a)

# 使用標準化將數據縮放為均值為 0 和標準差為 1 的數據
print(preprocessing.scale(a))

# 生成分類數據集，包含 300 個樣本，2 個特徵
# n_redundant：冗餘特徵的數量，n_informative：資訊特徵的數量
# random_state：隨機種子，n_clusters_per_class：每個類別的簇數
# scale：特徵數據的放大比例
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)

# 使用散點圖可視化生成的分類數據
plt.scatter(X[:, 0], X[:, 1], c=y)  # c=y 用於根據標籤顯示不同顏色
plt.show()  # 顯示散點圖

# 對生成的數據進行標準化處理，使每個特徵的均值為 0，標準差為 1
X = preprocessing.scale(X)

# 將數據集分割為訓練集和測試集，測試集佔 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# 初始化支持向量機分類器
clf = SVC()

# 使用訓練集來訓練支持向量機分類器
clf.fit(X_train, y_train)

# 評估模型在測試集上的準確度
print(clf.score(X_test, y_test))  # 打印模型的準確度

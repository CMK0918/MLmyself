# 從 scikit-learn 中導入必要的庫
from sklearn import datasets  # 用於加載資料集（此處使用鳶尾花資料集）
from sklearn.model_selection import train_test_split  # 用於將資料集分為訓練集和測試集
from sklearn.neighbors import KNeighborsClassifier  # K 最近鄰分類器（KNN）
from sklearn.metrics import accuracy_score # 用來評估模型的準確率

# 載入鳶尾花資料集
iris = datasets.load_iris()

# 從資料集中提取特徵（X）和標籤（y）
iris_X = iris.data  # 特徵：鳶尾花的各項測量數據
iris_y = iris.target  # 標籤：鳶尾花的種類（0, 1, 2）

# 可選：可以打印出前兩行特徵資料和所有標籤，了解資料結構
# print(iris_X[:2, :])  # 打印出前兩行的特徵資料
# print(iris_y)  # 打印出所有標籤資料

# 將資料集分為訓練集和測試集
# test_size=0.3 表示 30% 的資料用於測試，70% 用於訓練 同時打亂數據集
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

# 可選：可以打印出訓練集中的標籤，檢查標籤的分佈
# print(y_train)

# 初始化 K 最近鄰分類器（KNN）
knn = KNeighborsClassifier()

# 使用訓練資料（特徵和標籤）來訓練 KNN 模型
knn.fit(X_train, y_train)

# 使用訓練好的 KNN 模型來預測測試資料的標籤
y_pred = knn.predict(X_test)
print("預測標籤\n", y_pred)  # 打印出測試資料的預測標籤

# 打印出測試資料的真實標籤，以與預測結果進行對比
print("真實標籤\n", y_test)  # 打印出測試資料的真實標籤


accuracy = accuracy_score(y_test, y_pred)  # 計算測試集預測的準確率
print("模型的準確率", accuracy * 100,"%")  # 輸出模型的準確率
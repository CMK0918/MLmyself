import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 实现欧几里得距离函数
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 自定义KNN分类器
class KNeighborsClassifierCustom:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        # 训练阶段只是存储训练数据
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # 对每个测试样本进行预测
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # 计算测试样本与每个训练样本的距离
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 找到最近的K个邻居
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 通过投票确定类别（即选择出现次数最多的标签）
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# 测试自定义的KNN分类器
iris = load_iris()
X = iris.data
y = iris.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 实例化自定义KNN分类器并进行训练
knn_custom = KNeighborsClassifierCustom(n_neighbors=3)
knn_custom.fit(X_train, y_train)

# 使用测试集进行预测
y_pred_custom = knn_custom.predict(X_test)

# 计算准确率
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print("自定义KNN分类器的准确率:", accuracy_custom )




from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.neighbors import KNeighborsClassifier

# 1. 加載鳶尾花數據集
iris = load_iris()
X = iris.data
y = iris.target

# 3. 分割訓練集70%和測試集30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 4. KNN模型
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)

# 定義 KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
knn_cross_score = cross_val_score(knn_model,X_train,y_train,cv=kfold).mean()
knn_train_score = knn_model.score(X_train,y_train)
knn_test_score = knn_model.score(X_test,y_test)
print("train accuracy:",knn_train_score)
print("5 fold accuracy: ",knn_cross_score)
print("test accuracy: ",knn_test_score)

#====================================================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. 加載鳶尾花數據集
iris = load_iris()
X = iris.data
y = iris.target

# 3. 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 4. 隨機森林模型
rf_model = RandomForestClassifier(max_depth=3,random_state=0)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# 定義 KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
rf_cross_score = cross_val_score(rf_model,X_train,y_train,cv=kfold).mean()
rf_train_score = rf_model.score(X_train,y_train)
rf_test_score = rf_model.score(X_test,y_test)
print("train accuracy:",rf_train_score)
print("5 fold accuracy: ",rf_cross_score)
print("test accuracy: ",rf_test_score)

#====================================================================================

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 設置隨機種子
tf.random.set_seed(0)

# 載入數據集
iris = load_iris()
X = iris.data
y = to_categorical(iris.target, 3)

# 構建 MLP 模型函數
def MLP_model():
    model = Sequential()
    model.add(Dense(16, input_dim=4, activation='relu'))  # 增加隱藏層神經元
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 分割數據集為訓練集和測試集 (70%用於訓練，30%用於測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 在訓練集上進行 5 折交叉驗證
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

cross_scores = []
for train_index, val_index in kfold.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = MLP_model()
    model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=5, verbose=0)

    # 在驗證集上評估
    loss, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    cross_scores.append(accuracy)

# 輸出 5 折交叉驗證的平均準確率
mlp_cross_score = np.mean(cross_scores)
score = model.evaluate(X_train, y_train, verbose=0)
mlp_train_score = score[1]
score = model.evaluate(X_test, y_test, verbose=0)
mlp_test_score = score[1]
print("train accuracy: ",mlp_train_score)
print("5 fold accuracy: ",mlp_cross_score)
print("test accuracy: ",mlp_test_score,)

#====================================================================================

import pandas as pd
df = pd.DataFrame({
      "train accuracy":[knn_train_score,rf_train_score,mlp_train_score],
      "5 fold accuracy":[knn_cross_score,rf_cross_score,mlp_cross_score],
      "test accuracy":[knn_test_score,rf_test_score,mlp_test_score]
},index=["KNN","RandomForest","MLP"])

df
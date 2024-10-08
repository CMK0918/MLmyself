# 導入必要的庫
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加載鳶尾花數據集
iris = load_iris()
X = iris.data
y = iris.target

# 特徵標準化
X = StandardScaler().fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 2. 定義 10 折交叉驗證
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 3. 初始化變量來儲存每折的結果
scores = []

# 4. 手動進行 K 折交叉驗證
for train_idx, validation_idx in kfold.split(X_train, y_train):
    
    # 分別劃分訓練集和驗證集
    X_train_fold, X_validation = X_train[train_idx], X_train[validation_idx]
    y_train_fold, y_validation = y_train[train_idx], y_train[validation_idx]    

    # 構建模型（每次折疊都構建新的模型）
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),  # 明確指定輸入層的形狀
            tf.keras.layers.Dense(16, activation="relu"),  # 隱藏層
            tf.keras.layers.Dense(3, activation="softmax")  # 輸出層，對應3個類別
        ]
    )
    
    # 編譯模型
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    # 訓練模型
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=5, verbose=0, validation_data=(X_validation, y_validation), callbacks=[early_stopping])
    
    # 評估模型在驗證集上的性能
    score = model.evaluate(X_validation, y_validation, verbose=0)
    
    # 儲存該折疊的準確率
    scores.append(score[1])

# 5. 輸出 10 折交叉驗證的平均準確率與標準差
print(f"Cross-Validation Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")

# 6. 評估模型在測試集上的性能

score = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy: ", score[1])

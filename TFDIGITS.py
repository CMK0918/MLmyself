import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# 載入手寫數字資料集
dataset=datasets.load_digits()
x=dataset.data
y=dataset.target

# 觀察資料集
print("Features Shape: ", x.shape)
print("Target Shape: ", y.shape)
print(x[:2])
print(y[:5])

# one-hot-encoding
y=pd.get_dummies(y)
print(y[:5])

# 正規化特徵
x=x.astype("float32")/16.0

# 分訓練集80% 驗證集10% 測試集10%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_test,x_validation,y_test,y_validation=train_test_split(x_test,y_test,test_size=0.5,random_state=0)
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)

# 使用 Sequential API 構建模型

# 模型設置
model=tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64,activation="relu"), # input layer 8*8 pixels
        tf.keras.layers.Dense(128,activation="relu"), # hidden layer
        tf.keras.layers.Dropout(0.2),         # 加入適當dropout層可防止過擬和
        tf.keras.layers.Dense(10,activation="softmax") # output layer
    ]
)

# 編譯模型
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# 訓練模型
early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 早停機制
history=model.fit(x_train, y_train, epochs=200, batch_size=16, validation_data=(x_validation, y_validation), verbose=2, callbacks=[early_stopping])

# 評估模型
model.evaluate(x_test, y_test, batch_size=16, verbose=2)

# 將模型訓練結果存成表格形式
history_dict=pd.DataFrame(history.history)
print(history_dict)
epochs=range(1,len(history_dict["loss"])+1)
train_loss=history_dict["loss"]
validation_loss=history_dict["val_loss"]
train_accuracy=history_dict["accuracy"]
validation_accuracy=history_dict["val_accuracy"]

# 畫loss圖
plt.plot(epochs, train_loss, color="red", label="train loss")
plt.plot(epochs, validation_loss, color="black", label="validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("train and validation loss")
plt.legend()
plt.show()

# 畫accuracy圖
plt.plot(epochs, train_accuracy, color="red", label="train accuracy")
plt.plot(epochs, validation_accuracy, color="black", label="validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("train and validation accuracy")
plt.legend()
plt.show()

# 模型架構
model.summary()
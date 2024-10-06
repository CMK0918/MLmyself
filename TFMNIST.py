import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 用GPU加速
physical_device=tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0],True)

# 載入資料看資料形狀
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(y_train.shape)

# 觀察資料
plt.subplot(221)
plt.imshow(x_train[0],cmap="gray")
plt.subplot(222)
plt.imshow(x_train[1],cmap="gray")
plt.subplot(223)
plt.imshow(x_test[0],cmap="gray")
plt.subplot(224)
plt.imshow(x_test[1],cmap="gray")
plt.show()

# one-hot-encoding
y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)
print(y_train[:5])

# 展平資料
x_train=x_train.reshape(-1,28*28).astype("float32")/255.0
x_test=x_test.reshape(-1,28*28).astype("float32")/255.0
print(x_train.shape)
print(x_test.shape)

# 使用 Sequential API 構建模型

# 模型設置
model=tf.keras.Sequential(
    [
        tf.keras.layers.Dense(784,activation="relu"), # input layer 28*28 pixels
        tf.keras.layers.Dense(128,activation="relu"), # hidden layer
        tf.keras.layers.Dropout(0.3),         # 加入適當dropout層可防止過擬和
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
early_stopping=early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # 早停機制
history=model.fit(x_train, y_train, batch_size=200, epochs=50, validation_split=0.2, verbose=2, callbacks=[early_stopping])

# 評估模型
model.evaluate(x_test, y_test, batch_size=200, verbose=2)

# 將模型訓練結果存成表格形式
history_dict=pd.DataFrame(history.history)
print(history_dict)

# 提取畫圖所需參數
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 載入加州房價指數資料集
dataset=datasets.fetch_california_housing()
X=dataset.data
y=dataset.target

# 分割資料集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# 建立線性模型
model_LR=LinearRegression()
model_LR.fit(X_train,y_train)

# 評估模型準確性
print("線性回歸訓練集準確性R^2: ", model_LR.score(X_train,y_train))
print("線性回歸測試集準確性R^2: ", model_LR.score(X_test,y_test))
y_pred=model_LR.predict(X_test)
print("線性回歸MSE: ",mean_squared_error(y_test,y_pred))

plt.figure(1)
plt.plot(range(len(y_test)), sorted(y_test), color="black", label="test data")
plt.plot(range(len(y_pred)), sorted(y_pred), color="red", label="predict data")
plt.legend()
plt.show(block=False)


print("==============================================")

# 建立隨機森林回歸模型
model_RF=RandomForestRegressor(n_estimators=200,max_depth=24)
model_RF.fit(X_train,y_train)

# 評估模型性能
print("森林回歸訓練集準確性R^2: ", model_RF.score(X_train,y_train))
print("森林回歸測試集準確性R^2: ", model_RF.score(X_test,y_test))
y_pred=model_RF.predict(X_test)
print("森林回歸MSE: ",mean_squared_error(y_test,y_pred))

plt.figure(2)
plt.plot(range(len(y_test)), sorted(y_test), color="black", label="test data")
plt.plot(range(len(y_pred)), sorted(y_pred), color="red", label="predict data")
plt.legend()
plt.show()


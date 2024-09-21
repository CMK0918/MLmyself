import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 建立資料集
X, y = make_regression(n_samples=100, n_features=1, noise=15)
plt.figure(1) # 創建圖片視窗1
plt.title("sample set")
plt.scatter(X,y)
plt.show(block=False) # 允許多圖片視窗顯示

#  分割資料集 訓練集: 70%  測試集 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 建立 simple linear regression model      y=w1*x+w0
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)                # 用訓練集去 fit model
y_pred = regr.predict(X_test)             # 用測試集去 predict

# plt.scatter(X_train, y_train, color="black", label="train set") # 訓練結果
plt.figure(2)
plt.scatter(X_test, y_test, color="red", label="test set") # 測試結果
plt.plot(X_test, y_pred, color="blue", label="pred set") # 預測結果
plt.title("linear regression predict")
plt.legend()
plt.show(block=False)

w_0 = regr.intercept_
w_1 = regr.coef_

print("w0 截距", w_0)
print("w1 係數", w_1)

# 評估準確率 (若訓練測試差太多則為 overfitting)
print("訓練集模型準確度",regr.score(X_train, y_train)) # 訓練集準確度
print("測試集模型準確度",regr.score(X_test, y_test)) # 測試集準確度


# Gradient Decent 梯度下降法 自建模型

# 參數
alpha = 0.001 # learning rate
repeats = 1000 # 迭代次數

# 初始變數
w0 = 0 
w1 = 0
errors = []
points = []

for j in range(repeats):                    # y(x)=wo+w1x
    error_sum = 0                           # CF=sum(y-y(x))^2    
    squared_error_sum = 0                   # w0=w0+alpha*sum(y-y(x))
    error_sum_x = 0                         # w1=w1+alpha*sum(y-y(x))*x
    for i in range(len(X_train)):
        predict = w0 + (X_train[i] * w1)
        squared_error_sum = squared_error_sum + (y_train[i] - predict)**2
        error_sum = error_sum + y_train[i] - predict
        error_sum_x= error_sum_x + (y_train[i] -predict) * X_train[i]
    w0 = w0 + (alpha * error_sum)
    w1 = w1 + (alpha * error_sum_x)
    errors.append(squared_error_sum/len(X_train))

print("w0 截距", w0)
print("w1 係數", w1)

predicts = []
mean_error = 0
for i in range(len(X_test)):
    predict = w0 + (X_test[i] * w1)
    predicts.append(predict)

plt.figure(3)
plt.plot(X_test, predicts, color="blue", label="predict set")
plt.scatter(X_test, y_test , color="red", label="test set" )
plt.title("gradient decent redression predict")
plt.legend()
plt.show()






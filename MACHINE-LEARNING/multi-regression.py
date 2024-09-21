import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 創建資料集 5個特徵項
X, y = make_regression(n_samples=100, n_features=5, noise=20)

# 分割資料集  訓練集:70%  測試集:30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 建立多元迴歸模型
regr = linear_model.LinearRegression()  # y=w0+w1*x1+w2*x2+w3*x3+w4*x4+w5*x5
regr.fit(X_train, y_train)

print("截距 w0: ",regr.intercept_) # 截距
print("係數 w1 w2 w3 w4 w5: ",regr.coef_) # 係數
print("訓練集準確度",regr.score(X_train,y_train))
print("測試集準確度", regr.score(X_test,y_test))


# 創建資料集 2個特徵項
size = [5,10,12,14,18,30,33,55,65,80,100,150]                      # 房子大小
distance = [50,20,70,100,200,150,30,50,70,35,40,20]                # 距離市中心距離
price = [300,400,450,800,1200,1400,2000,2500,2800,3000,3500,9000]  # 房價

df = pd.DataFrame({
    "X1": size,
    "X2": distance,
    "y": price
})

X = df[["X1","X2"]]
y = df[["y"]]

# 建立多元迴歸模型
regr = linear_model.LinearRegression()
regr.fit(X, y)

print("訓練模型準確度", regr.score(X,y))
print("截距項 w0: ", regr.intercept_)
print("係數項 w1,w2: ", regr.coef_)
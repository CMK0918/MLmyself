import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 創建多特徵資料集做展示
X, y = make_regression(n_samples=100, n_features=10, noise=10)

# 分割資料集 訓練集:70% 測試集:30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# 建立多元迴歸模型
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# 評估準確度
print("多元迴歸訓練集準確度", regr.score(X_train, y_train))
print("多元迴歸測試集準確度", regr.score(X_test, y_test))
print("===================我是分隔線=======================")


# 建立 Lasso regression 模型
clf_lasso = linear_model.Lasso(alpha=0.5)
clf_lasso.fit(X_train, y_train)

# 評估準確度
print("Lasso訓練集準確度", clf_lasso.score(X_train, y_train)) # 訓練集準確度
print("Lasso測試集準確度", clf_lasso.score(X_test, y_test))
print("===================我是分隔線=======================")


# 建立 Ridge Regression 模型
clf_ridge = linear_model.Ridge(alpha=0.5)
clf_ridge.fit(X_train , y_train)

# 評估準確度
print("Ridge訓練集準確度", clf_ridge.score(X_train, y_train))
print("Ridge測試集準確度", clf_ridge.score(X_test, y_test))
print("===================我是分隔線=======================")


# 建立多項式迴歸用Ridee regression正規化模型
model = make_pipeline(PolynomialFeatures(4), linear_model.Ridge())
model.fit(X,y)
print("模型準確度", model.score(X,y))
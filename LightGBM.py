import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import datasets
import matplotlib.pyplot as plt

# 數據準備
df = datasets.fetch_california_housing()
X = df.data
y = df.target

# print(X.shape)
# print(y.shape)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)

# GBR model built
GBR = GradientBoostingRegressor(n_estimators=150 ,learning_rate=0.25, random_state=0)
GBR.fit(X_train,y_train)

# # 調參
# from time import time
# for i in range(50,550,50):
#     start = time()
#     GBR = GradientBoostingRegressor(n_estimators=i, random_state=0)
#     GBR.fit(X_train, y_train)
#     print("estimator:", i, " R^2 5fold:", cross_val_score(GBR, X_train, y_train, cv=5).mean(), " time:", time()-start)

# # 調參
# from time import time
# for i in np.linspace(0.01, 0.2, 10):
#     start = time()
#     GBR = GradientBoostingRegressor(learning_rate=i, n_estimators=150, random_state=0)
#     GBR.fit(X_train, y_train)
#     print("estimator:", i, " R^2 5fold:", cross_val_score(GBR, X_train, y_train, cv=5).mean(), " time:", time()-start)


# model evaluate
y_pred = GBR.predict(X_test)
mae_GBR = mean_absolute_error(y_test, y_pred)
train_score_GBR = GBR.score(X_train, y_train)
test_score_GBR = GBR.score(X_test, y_test)
cross_score_GBR = cross_val_score(GBR, X_train, y_train, cv=5).mean()

print("train R^2: ", train_score_GBR)
print("5 fold R^2: ", cross_score_GBR)
print("test R^2: ", test_score_GBR)
print("MAE: ", mae_GBR)
#==============================================================================================================
# lightGBM model build

from lightgbm import LGBMRegressor

LGBR = LGBMRegressor()
LGBR.fit(X_train,y_train)

# model evaluate
y_pred = LGBR.predict(X_test)
mae_XGBR = mean_absolute_error(y_test, y_pred)
train_score_XGBR = LGBR.score(X_train, y_train)
test_score_XGBR = LGBR.score(X_test, y_test)
cross_score_XGBR = cross_val_score(LGBR, X_train, y_train, cv=5).mean()

print("train R^2: ", train_score_XGBR)
print("5 fold R^2: ", cross_score_XGBR)
print("test R^2: ", test_score_XGBR)
print("MAE: ", mae_XGBR)

#==============================================================================================================
# xgboost model build

from xgboost import XGBRegressor

XGBR = XGBRegressor()
XGBR.fit(X_train,y_train)

# model evaluate
y_pred = XGBR.predict(X_test)
mae_XGBR = mean_absolute_error(y_test, y_pred)
train_score_XGBR = XGBR.score(X_train, y_train)
test_score_XGBR = XGBR.score(X_test, y_test)
cross_score_XGBR = cross_val_score(XGBR, X_train, y_train, cv=5).mean()

print("train R^2: ", train_score_XGBR)
print("5 fold R^2: ", cross_score_XGBR)
print("test R^2: ", test_score_XGBR)
print("MAE: ", mae_XGBR)
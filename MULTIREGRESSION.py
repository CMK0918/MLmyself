import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 載入資料集
dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:, 0:4]
y=dataset.iloc[:, 4]
# print(dataset)
# print(X)
# print(y)
# print(X["State"].unique())

# one hot encoding
pd.get_dummies(X["State"])
statedump=pd.get_dummies(X["State"],drop_first=True) # 01 10 00 就可代表三個州降低共線性問題
# print(statedump)
X=X.drop("State",axis=1)
# print(X)
X=pd.concat([X,statedump],axis=1)
# print(X)
X=StandardScaler().fit_transform(X)
print(X)

# 分割資料集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# 建模型 多元線性迴歸
model=LinearRegression()
model.fit(X_train,y_train)

# 評估模型準確度
print("訓練集準確度R^2: ",model.score(X_train,y_train))
print("測試集準確度R^2: ",model.score(X_test,y_test))


print("截距項: ",model.intercept_)
print("係數項: ",model.coef_)

y_pred=model.predict(X_test)
print("真實profit: ",y_test)
print("預測profit: ",y_pred)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Polymial Regression 多項式迴歸

# 創建樣本點
size = [5,10,12,14,18,30,33,55,65,80,100,150]
price = [300,400,450,800,1200,1400,2000,2500,2800,3000,3500,9000]

plt.figure(1) # 創建圖1視窗
plt.scatter(size,price)
plt.title("sample point")
plt.show(block=False) # 允許多圖視窗顯示

df = pd.DataFrame({
    "X":size,
    "y":price
})
X = df[["X"]]
y = df[["y"]]

# 創建多項式迴歸模型 3次多項式線性迴歸
model = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(2) # 創建圖2視窗
plt.scatter(X, y, label="sample point", color="blue")
plt.plot(X, y_pred, label="predict line", color="red")
plt.legend()
plt.title("polynomial regression predict model")
plt.show(block=False) # 允許多圖視窗顯示


# 用迴圈確認使用幾次多項式迴歸較合適
scores = []
colors = ["green", "purple", "gold", "blue", "black"]

for count,degree in enumerate([1,2,3,4,5]):
    model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
    model.fit(X, y)
    scores.append(model.score(X, y))
    y_pred = model.predict(X)
    
    plt.figure(3) #創建圖3視窗
    plt.plot(X, y_pred, color=colors[count], label="degree %d" %degree)

plt.legend(loc=2)
plt.title("polynomial degree")
plt.show()
print(scores)
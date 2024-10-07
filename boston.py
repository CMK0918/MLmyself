import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# 載入資料
df=pd.read_csv("boston_house_prices.csv")
#print(df)
X=(df.iloc[1:, :13]).astype("float32")
y=(df.iloc[1:, 13]).astype("float32")


# 正規化
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X)

# 分割資料
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# 建立線性回歸模型
model=LinearRegression()
model.fit(X_train,y_train)

# 評估模型準確度
y_pred=model.predict(X_test)
print("測試集R^2: ", model.score(X_test,y_test))
print("每折驗證R^2: ", cross_val_score(model,X,y,cv=5))
print("交叉驗證R^2: ", cross_val_score(model,X,y,cv=5).mean())
print("MSE:", metrics.mean_squared_error(y_pred,y_test))
print("5 fold MSE: ", -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
print("cross MSE: ", -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())

error=y_pred-y_test
plt.scatter(range(len(y_test)),error)
plt.xlabel("sample")
plt.ylabel("error")
plt.title("LinearRegression") 
plt.show()

#====================================================================

# 建立隨機森林回歸模型
model_RF=RandomForestRegressor()
model_RF.fit(X_train,y_train)

# 評估模型準確度
y_pred=model_RF.predict(X_test)
print("測試集R^2: ", model_RF.score(X_test,y_test))
print("每折驗證R^2: ", cross_val_score(model_RF,X,y,cv=5))
print("交叉驗證R^2: ", cross_val_score(model_RF,X,y,cv=5).mean())
print("MSE:", metrics.mean_squared_error(y_pred,y_test))
print("5 fold MSE: ", -cross_val_score(model_RF, X, y, cv=5, scoring='neg_mean_squared_error'))
print("cross MSE: ", -cross_val_score(model_RF, X, y, cv=5, scoring='neg_mean_squared_error').mean())

# 評估模型準確度
error=y_pred-y_test.astype(float)
plt.scatter(range(len(y_test)),error) 
plt.xlabel("sample")
plt.ylabel("error")
plt.title("RandomForestRegression") 
plt.show()


# # 網格搜索法交差驗證找合適超參數
# from sklearn.model_selection import GridSearchCV

# # 樹數目、樹深度、最大特徵數
# param_grid = {
#     'n_estimators':[20,50,70,100,120,160,200],
#     'max_depth':[3,5,7,9],
#     'max_features':[0.1,0.3,0.5,0.6,0.7] 
# }

# # 隨機森林回歸器模型
# RF = RandomForestRegressor()

# # 網格搜尋回歸器模型
# grid = GridSearchCV(RF, param_grid=param_grid, cv=5)  

# # 訓練網格回歸器模型
# grid.fit(X_train,y_train)

# # 找出最優參數及交叉驗證評分
# print(grid.best_params_)
# print("最優評分:",grid.best_score_)



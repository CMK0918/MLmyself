import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 載入資料
df = pd.read_csv("boston_house_prices.csv")
X = (df.iloc[:, :13]).astype("float32")
y = (df.iloc[:, 13]).astype("float32")

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#====================================================================

# 建立隨機森林回歸模型
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# 評估模型準確度
y_pred_rf = model.predict(X_test)

print("隨機森林訓練集R^2: ", model.score(X_train, y_train))
print("隨機森林5折驗證R^2: ", cross_val_score(model, X_train, y_train, cv=5).mean())
print("隨機森林測試集R^2: ", model.score(X_test, y_test))
print("隨機森林MSE:", metrics.mean_squared_error(y_pred_rf, y_test))
print("隨機森林MAE:", metrics.mean_absolute_error(y_pred_rf, y_test))  # 計算MAE

# 評估模型準確度
error_rf = y_pred_rf - y_test.astype(float)
plt.figure(2)
plt.scatter(range(len(y_test)), error_rf)
plt.ylim(-30, 30)
plt.xlabel("sample")
plt.ylabel("error")
plt.title("RandomForestRegression")
plt.show(block=False)

#====================================================================

# 建立梯度提升樹回歸模型
model = GradientBoostingRegressor(random_state=0)
model.fit(X_train, y_train)

# 評估模型準確度
y_pred_gbr = model.predict(X_test)

print("梯度提升樹訓練集R^2: ", model.score(X_train, y_train))
print("梯度提升樹5折驗證R^2: ", cross_val_score(model, X_train, y_train, cv=5).mean())
print("梯度提升樹測試集R^2: ", model.score(X_test, y_test))
print("梯度提升樹MSE:", metrics.mean_squared_error(y_pred_gbr, y_test))
print("梯度提升樹MAE:", metrics.mean_absolute_error(y_pred_gbr, y_test))  # 計算MAE

# 評估模型準確度
error_gbr = y_pred_gbr - y_test.astype(float)
plt.figure(3)
plt.scatter(range(len(y_test)), error_gbr)
plt.ylim(-30, 30)
plt.xlabel("sample")
plt.ylabel("error")
plt.title("GradientBoostingRegressor")
plt.show(block=False)

#====================================================================

# 使用加權平均進行模型融合
weight_rf = 0.05  # 隨機森林的權重
weight_gbr = 0.95  # 梯度提升樹的權重

# 加權平均預測
y_pred_weighted = weight_rf * y_pred_rf + weight_gbr * y_pred_gbr

# 評估加權融合模型
weighted_r2 = metrics.r2_score(y_test, y_pred_weighted)
weighted_mse = metrics.mean_squared_error(y_test, y_pred_weighted)
weighted_mae = metrics.mean_absolute_error(y_test, y_pred_weighted)  # 計算MAE

print("加權融合模型測試集R^2:", weighted_r2)
print("加權融合模型MSE:", weighted_mse)
print("加權融合模型MAE:", weighted_mae)  # 顯示MAE

# 可視化加權融合模型誤差
error_weighted = y_pred_weighted - y_test
plt.figure()
plt.scatter(range(len(y_test)), error_weighted)
plt.ylim(-30, 30)
plt.xlabel("sample")
plt.ylabel("error")
plt.title("Weighted Average Model")
plt.show()



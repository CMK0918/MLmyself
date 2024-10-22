import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import datasets
from model_construct import rf_model, gbr_model, lgbm_model, print_score

# 數據準備
df = datasets.fetch_california_housing()
X = df.data
y = df.target

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 建模計算
rf_model_result = rf_model(treenumber=150, depth=15)
lgbm_model_result = lgbm_model()
gbr_model_result = gbr_model(treenumber=150, learnrate=0.2)

# 打印評估分數
print_score(rf_model_result)
print_score(lgbm_model_result)
print_score(gbr_model_result)

# 模型融合
# Weighted Average
weight_gbr = 0.09  # GBDT
weight_rf = 0.01  # RF
weight_lgbr = 0.9  # LGBM

# 加權平均預測
y_pred_test_weight = weight_rf * rf_model_result[4] + weight_gbr * gbr_model_result[4] + weight_lgbr * lgbm_model_result[4]

# 評估加權融合模型
weighted_test_r2 = metrics.r2_score(y_test, y_pred_test_weight)
weighted_mse = metrics.mean_squared_error(y_test, y_pred_test_weight)  # 計算MSE

print("Average test R2:", weighted_test_r2)
print("Average MAE:", weighted_mse)  # 顯示MSE
#=================================================================================================================
df = pd.DataFrame({
    "R2":[gbr_model_result[2], lgbm_model_result[2], rf_model_result[2], weighted_test_r2],
    "MSE":[gbr_model_result[3], lgbm_model_result[3], rf_model_result[3], weighted_mse]
},index=["GBDT","LGBM","RF","Avg"])

print(df)
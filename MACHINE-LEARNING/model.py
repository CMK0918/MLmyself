from sklearn import datasets  # 從 scikit-learn 中導入 datasets 模組，用於加載內建的資料集
from sklearn.linear_model import LinearRegression  # 從 scikit-learn 中導入 LinearRegression 類，用於執行線性回歸

# 載入乳腺癌資料集
loaded_data = datasets.load_breast_cancer()

# 提取特徵資料（X）和目標資料（y）
data_X = loaded_data.data  # 特徵資料：描述乳腺癌樣本的各種測量值
data_y = loaded_data.target  # 目標資料：樣本的分類標籤（0 代表良性，1 代表惡性）

# 初始化線性回歸模型
model = LinearRegression()

# 使用特徵資料（X）和目標資料（y）來訓練線性回歸模型
model.fit(data_X, data_y)

# 使用訓練好的模型對前四個樣本進行預測
print(model.predict(data_X[:4, :]))  # 打印出模型對前四個樣本的預測結果

# 打印模型的係數，表示每個特徵對預測的影響程度
print(model.coef_)  # 顯示各個特徵的係數，例如，0.1 表示某個特徵對預測的影響

# 打印模型的截距，當所有特徵值為零時的預測值
print(model.intercept_)  # 顯示模型的截距，例如，0.3

# 打印模型的參數，如正則化參數等
print(model.get_params())  # 顯示模型的各種參數，例如，是否使用正則化等

# 打印模型的 R^2 決定係數，衡量模型對資料的擬合程度
print(model.score(data_X, data_y))  # R^2 決定係數，表示模型解釋了目標變量變異的百分比

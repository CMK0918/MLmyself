# 從 scikit-learn 中導入需要的庫
from sklearn import datasets  # 用於加載內建的數據集
from sklearn.linear_model import LinearRegression  # 用於執行線性回歸
import matplotlib.pyplot as plt  # 用於資料視覺化

# 載入乳腺癌資料集，該資料集內建於 scikit-learn 中
loaded_data = datasets.load_breast_cancer()

# 將特徵資料（X）和標籤資料（y）分別提取出來
data_x = loaded_data.data  # 特徵資料：描述癌症的不同測量值
data_y = loaded_data.target  # 標籤資料：0 代表良性，1 代表惡性

# 初始化線性回歸模型
model = LinearRegression()

# 使用線性回歸模型來擬合資料，學習 X 和 y 之間的關係
model.fit(data_x, data_y)

# 使用已訓練好的模型來預測前四個樣本的標籤
print(model.predict(data_x[:4, :]))  # 打印出預測結果
print(data_y[:4])  # 打印出前四個樣本的真實標籤，供比較

# 創建一個隨機生成的回歸資料集
# n_samples: 資料點數量, n_features: 特徵數量, n_targets: 目標變量的數量, noise: 添加的隨機噪音
x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)

# 使用散點圖來視覺化隨機生成的回歸資料集
plt.scatter(x, y)
plt.show()  # 顯示散點圖
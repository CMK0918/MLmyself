from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 載入數據集
iris=datasets.load_iris()
X=iris.data
y=iris.target

# 分割資料集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# 建構隨機森林模型
rfc=RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=50,min_samples_leaf=5)
rfc.fit(X_train,y_train)

# y_predict=rfc.predict(X_test)
# print("測試結果:",y_test)
# print("預測結果:",y_predict)

# 評估模型準確度
print("測試集準確度:",rfc.score(X_test,y_test))


print("=======================================")


# Feature Importance
imp=rfc.feature_importances_
print(imp)
names=iris.feature_names
print(names)

zip(imp,names)
imp,names=zip(*sorted(zip(imp,names)))
plt.barh(range(len(names)),imp,align="center")
plt.yticks(range(len(names)),names)
plt.xlabel("Importance of Features")
plt.ylabel("Features")
plt.title("Importance of Each Feature")
plt.show()

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 載入乳癌資料集
data=datasets.load_breast_cancer() 
# print(data)
X=data.data
y=data.target

# 分割資料集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# 決策樹模型
model=tree.DecisionTreeClassifier(max_depth=2)
model.fit(X_train,y_train)

# 評估模型準確度
print("訓練集評分",model.score(X_train,y_train))
print("測試集評分",model.score(X_test,y_test))
score=cross_val_score(model,X,y,cv=10) # 資料集切10等分 9訓練1測試做10次評分
print("交叉驗證平均得分",score.mean())

# 繪製決策樹
feature_names=data.feature_names
target_names=data.target_names
plt.figure(figsize=(9,9))
plot_tree(model,feature_names=feature_names,class_names=target_names,filled=True)
plt.show()

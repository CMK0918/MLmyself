from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree

# 載入乳癌資料集
data=datasets.load_breast_cancer()
X=data.data
y=data.target

# 分割資料集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# 建立隨機森林模型
model=RandomForestClassifier(n_estimators=5,max_depth=2) # 設定5棵樹,2層深度
model.fit(X_train,y_train)

# 評估模型準確度
print("訓練集分數",model.score(X_train,y_train))
print("測試集分數",model.score(X_test,y_test))
score=cross_val_score(model,X,y,cv=10)
print("交叉驗證分數",score.mean())

# 繪製森林圖
feature_names=data.feature_names
target_names=data.target_names

plt.figure(figsize=(128,64))
for i, DecisionTree in enumerate(model.estimators_):
    plt.subplot(1, 5, i+1)
    tree.plot_tree(DecisionTree,feature_names=feature_names,class_names=target_names,filled=True)
plt.savefig('forest.png')


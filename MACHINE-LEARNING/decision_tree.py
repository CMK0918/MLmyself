from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pydotplus
import pandas as pd
import numpy as np

# 載入資料集
iris = datasets.load_iris()
X = iris.data
y= iris.target

# 分類模型
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X,y)

# 準確度評估
print(clf.score(X,y))

# 決策樹模型輸出 流程圖 pdf 
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")

print("=============================我是分隔線=============================")
# 拆分訓練集與測試集  訓練集: 70 %   測試集: 30 %
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# 分類模型
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train,y_train)

# 準確度評估
print(clf.score(X_test,y_test))

print(y_test)
y_pred = clf.predict(X_test)
print(y_pred)

print("=============================我是分隔線=============================")
# 過度配適初步調整 改criterion 限制樹深度
clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=3)
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

print(y_test)
y_pred = clf.predict(X_test)
print(y_pred)

# 決策樹模型輸出 流程圖 pdf 
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris_gini_max3.pdf")
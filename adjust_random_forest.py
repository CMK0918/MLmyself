from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 載入資料集
data = datasets.load_digits()
# print(data)
X=data.data
# print(X)
y=data.target
# print(y)

# 建立模型 1.調整criterion  (測試出來entropy較佳)
# RF=RandomForestClassifier(random_state=66,n_jobs=-1)
# score=cross_val_score(RF,X,y,cv=10) # 分10個子集進行10次訓練(每次9個訓練集1個測試集)
# print("gini交叉驗證平均得分: ", score.mean())

RF=RandomForestClassifier(criterion="entropy",random_state=66,n_jobs=-1)
score=cross_val_score(RF,X,y,cv=10) # 分10個子集進行10次訓練(每次9個訓練集1個測試集)
print("entropy交叉驗證平均得分: ", score.mean())


# 建立模型 2.粗調整n_estimator參數 (測試出來140最優)
# ScoreAll=[]
# for i in range(10,200,10):
#     DT=RandomForestClassifier(n_estimators=i,criterion="entropy",random_state=66,n_jobs=-1)
#     score=cross_val_score(DT,X,y,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll=np.array(ScoreAll)

# max_score=np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0]
# print("最優棵數和最高得分",ScoreAll[max_score])
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.show()

# 建立模型 2.細調整n_estimator參數 (測試出來136最優)
# ScoreAll=[]
# for i in range(130,150):
#     DT=RandomForestClassifier(n_estimators=i,criterion="entropy",random_state=66,n_jobs=-1)
#     score=cross_val_score(DT,X,y,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll=np.array(ScoreAll)

# max_score=np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0]
# print("最優棵數和最高得分",ScoreAll[max_score])
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.show()


# 建立模型 3.調整max_depth參數 (測試出來10最優)
# ScoreAll=[]
# for i in range(1,30):
#     DT=RandomForestClassifier(max_depth=i,n_estimators=136,criterion="entropy",random_state=66,n_jobs=-1)
#     score=cross_val_score(DT,X,y,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll=np.array(ScoreAll)

# max_score=np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0]
# print("最優深度和最高得分",ScoreAll[max_score])
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.show()


# 建立模型 4.調整min_sample_split參數 (測試出來2最優)
# ScoreAll=[]
# for i in range(2,10):
#     DT=RandomForestClassifier(min_samples_split=i,max_depth=10,n_estimators=136,criterion="entropy",random_state=66,n_jobs=-1)
#     score=cross_val_score(DT,X,y,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll=np.array(ScoreAll)

# max_score=np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0]
# print("最優分支點樣本數和最高得分",ScoreAll[max_score])
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.show()


# 建立模型 5.調整min_sample_leaf參數 (測試出來1最優)
ScoreAll=[]
for i in range(1,10):
    DT=RandomForestClassifier(min_samples_leaf=i,min_samples_split=2,max_depth=10,n_estimators=136,criterion="entropy",random_state=66,n_jobs=-1)
    score=cross_val_score(DT,X,y,cv=10).mean()
    ScoreAll.append([i,score])
ScoreAll=np.array(ScoreAll)

max_score=np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0]
print("最優葉結點最少樣本數和最高得分",ScoreAll[max_score])
plt.figure(figsize=[20,5])
plt.plot(ScoreAll[:,0],ScoreAll[:,1])
plt.show()



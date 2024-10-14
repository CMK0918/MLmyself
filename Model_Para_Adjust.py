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
X = data.data
y = data.target
# print(data)
# print(X)
# print(y)

#===============================================超參數調整=====================================================#

# 建立模型 1.調整criterion  (測試出來entropy較佳)
# RF=RandomForestClassifier(random_state=66,n_jobs=-1)
# score=cross_val_score(RF,X,y,cv=10) # 分10個子集進行10次訓練(每次9個訓練集1個測試集)
# print("gini交叉驗證平均得分: ", score.mean())

# RF=RandomForestClassifier(criterion="entropy",random_state=66,n_jobs=-1)
# score=cross_val_score(RF,X,y,cv=10) # 分10個子集進行10次訓練(每次9個訓練集1個測試集)
# print("entropy交叉驗證平均得分: ", score.mean())


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

# 建立模型 2.細調整n_estimators參數 (測試出來136最優)
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


# 建立模型 4.調整min_samples_split參數 (測試出來2最優)
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


# 建立模型 5.調整min_samples_leaf參數 (測試出來1最優)
# ScoreAll=[]
# for i in range(1,10):
#     DT=RandomForestClassifier(min_samples_leaf=i,min_samples_split=2,max_depth=10,n_estimators=136,criterion="entropy",random_state=66,n_jobs=-1)
#     score=cross_val_score(DT,X,y,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll=np.array(ScoreAll)

# max_score=np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0]
# print("最優葉結點最少樣本數和最高得分",ScoreAll[max_score])
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.show()

# 建立模型 6.網格搜尋調整max_features參數 (測出來10%最優) 
# param_grid={"max_features":np.arange(0.1, 1)}
# DT=RandomForestClassifier(min_samples_leaf=1,min_samples_split=2,max_depth=10,n_estimators=136,criterion="entropy",random_state=66,n_jobs=-1)
# GS=GridSearchCV(DT,param_grid,cv=10)
# GS.fit(X,y)
# print("最優參數:",GS.best_params_)
# print("最優評分:",GS.best_score_)


# 比較調參數模型與預設模型準確度
DT=RandomForestClassifier(oob_score=True,max_features=0.1,min_samples_leaf=1,min_samples_split=2,max_depth=10,n_estimators=136,criterion="entropy",random_state=66,n_jobs=-1)
DT=DT.fit(X,y)
score=cross_val_score(DT,X,y,cv=10) # 分10個子集進行10次訓練(每次9個訓練集1個測試集)
print("調參交叉驗證平均得分: ", score.mean())
print("調參oob得分: ", DT.oob_score_)

DT=RandomForestClassifier(oob_score=True,random_state=66,n_jobs=-1)
DT=DT.fit(X,y)
score=cross_val_score(DT,X,y,cv=10) # 分10個子集進行10次訓練(每次9個訓練集1個測試集)
print("預設交叉驗證平均得分: ", score.mean())
print("預設oob得分: ", DT.oob_score_)

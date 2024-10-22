import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import datasets

# 數據準備
df = datasets.fetch_california_housing()
X = df.data
y = df.target

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# RandomForest model build

def rf_model(treenumber:int, depth:int) -> list:
    '''
    this function build RandmoForestRegressor model
    and return train R2, cross_validation R2, test R2, test MAE, y_pred 
    '''

    # initial result list
    result = []

    # model fit
    RF = RandomForestRegressor(n_estimators=treenumber, max_depth=depth, random_state=0)
    RF.fit(X_train,y_train)

    # model evaluate
    y_pred = RF.predict(X_test)
    mae_RF = mean_absolute_error(y_test, y_pred)
    train_score_RF = RF.score(X_train, y_train)
    test_score_RF = RF.score(X_test, y_test)
    cross_score_RF = cross_val_score(RF, X_train, y_train, cv=5).mean()

    # append result to the list
    result.append(train_score_RF)
    result.append(cross_score_RF)
    result.append(test_score_RF)
    result.append(mae_RF)
    result.append(y_pred)

    return result


def lgbm_model() ->list:
    '''
    this function build LGBMRegressor model
    and return train R2, cross_validation R2, test R2, test MAE, y_pred 
    '''

    # initial result list
    result = []

    # model fit
    LGBR = LGBMRegressor(random_state=0)
    LGBR.fit(X_train,y_train)

    # model evaluate
    y_pred_LGBR = LGBR.predict(X_test)
    mae_LGBR= mean_absolute_error(y_test, y_pred_LGBR)
    train_score_LGBR = LGBR.score(X_train, y_train)
    test_score_LGBR = LGBR.score(X_test, y_test)
    cross_score_LGBR = cross_val_score(LGBR, X_train, y_train, cv=5).mean()

    # append result to the list
    result.append(train_score_LGBR)
    result.append(cross_score_LGBR)
    result.append(test_score_LGBR)
    result.append(mae_LGBR)
    result.append(y_pred_LGBR)

    return result


def gbr_model(treenumber:int, learnrate:any) -> list:
    '''
    this function build GradientBoostingRegressor model
    and return train R2, cross_validation R2, test R2, test MAE, y_pred     
    '''
    # initial result list
    result = []

    # model fit
    GBR = GradientBoostingRegressor(n_estimators=treenumber ,learning_rate=learnrate, random_state=0)
    GBR.fit(X_train,y_train)

    # model evaluate
    y_pred_GBR = GBR.predict(X_test)
    mae_GBR = mean_absolute_error(y_test, y_pred_GBR)
    train_score_GBR = GBR.score(X_train, y_train)
    test_score_GBR = GBR.score(X_test, y_test)
    cross_score_GBR = cross_val_score(GBR, X_train, y_train, cv=5).mean()

    # append result to the list
    result.append(train_score_GBR)
    result.append(cross_score_GBR)
    result.append(test_score_GBR)
    result.append(mae_GBR)
    result.append(y_pred_GBR)

    return result


def print_score(modedl_result:list) -> None:
    '''
    input model_result list , then print evaluate score return None 
    '''
    print("train R2: ", modedl_result[0])
    print("5 fold R2: ", modedl_result[1])
    print("test R2: ", modedl_result[2])
    print("MAE: ", modedl_result[3])
    return




# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:57:03 2018

@author: Uvais Karni
"""
from statistics import mode
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

AllRuns=[]
rmae=[]
rmse=[]
rscore=[]
bestpara=[]

data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
label = data.SalePrice
data= data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

my_imputer = Imputer()
data = my_imputer.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.3, random_state=42)


def AutoRFR(x_train, x_test, y_train, y_test):
    #Auto starts
    rfc=RandomForestRegressor(random_state=42)
    param_grid = { 
        'n_estimators': [i for i in range(500,1000) if i % 100 == 0],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [i for i in range(3,9)]
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(x_train, y_train)
    #all run parameters
    AllRun=pd.DataFrame(CV_rfc.grid_scores_)
    #save the best parameters
    best=CV_rfc.best_params_
    #train for best Parameters
    rfc1=RandomForestRegressor(random_state=42, max_features=best['max_features'], n_estimators= best['n_estimators'], max_depth=best['max_depth'])
    rfc1.fit(x_train, y_train)
    #Predict the values
    pred=rfc1.predict(x_test)
    #generate Score
    rmae=mean_absolute_error(pred, y_test)
    rmse=mean_squared_error(pred, y_test)
    rscore=r2_score(pred,y_test)
    return AllRun,best,pred,rmae,rmse,rscore

def AutoXGBR(x_train, x_test, y_train, y_test):
    #Auto starts
    rfc=XGBRegressor(random_state=42)
    param_grid = {'min_child_weight':[4,5],
              'gamma':[i/10.0 for i in range(3,6)],
              'subsample':[i/10.0 for i in range(6,11)],
              'colsample_bytree':[i/10.0 for i in range(6,11)],
              'max_depth': [2,3,4]}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(x_train, y_train)
    #all run parameters
    AllRun=pd.DataFrame(CV_rfc.grid_scores_)
    #save the best parameters
    best=CV_rfc.best_params_
    #train for best Parameters
    rfc1=XGBRegressor(random_state=42, min_child_weight=best['min_child_weight'], gamma= best['gamma'], subsample=best['subsample'], colsample_bytree=best['colsample_bytree'], max_depth=best['max_depth'])
    rfc1.fit(x_train, y_train)
    #Predict the values
    pred=rfc1.predict(x_test)
    #generate Score
    rmae=mean_absolute_error(pred, y_test)
    rmse=mean_squared_error(pred, y_test)
    rscore=r2_score(pred,y_test)
    return AllRun,best,pred,rmae,rmse,rscore

def AutoLGBMR(x_train, x_test, y_train, y_test):
    #Auto starts
    rfc=lgb.LGBMRegressor(random_state=42)
    param_grid =  {
            'learning_rate': [i/1000 for i in range(1,103) if i%10 ==0],
            'num_leaves': [i for i in range(2,90)],
            'bagging_fraction': [i/10 for i in range(1,11)],
            'min_data_in_leaf': [i for i in range(1,90) if i%5 ==0]}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(x_train, y_train)
    #all run parameters
    AllRun=pd.DataFrame(CV_rfc.grid_scores_)
    #save the best parameters
    best=CV_rfc.best_params_
    #train for best Parameters
    rfc1=lgb.LGBMRegressor(random_state=42, learning_rate=best['learning_rate'], num_leaves= best['num_leaves'], bagging_fraction=best['bagging_fraction'], min_data_in_leaf=best['min_data_in_leaf'])
    rfc1.fit(x_train, y_train)
    #Predict the values
    pred=rfc1.predict(x_test)
    #generate Score
    rmae=mean_absolute_error(pred, y_test)
    rmse=mean_squared_error(pred, y_test)
    rscore=r2_score(pred,y_test)
    return AllRun,best,pred,rmae,rmse,rscore

#call the function AutoRFR
AllRun,best,pred,rmae_s,rmse_s,rscore_s=AutoRFR(x_train, x_test, y_train, y_test)
AllRuns.append(AllRun)
bestpara.append(best)
rmae.append(rmae_s)
rmse.append(rmse_s)
rscore.append(rscore_s)
#call the function AutoXGBR
AllRun,best,pred,rmae_s,rmse_s,rscore_s=AutoXGBR(x_train, x_test, y_train, y_test)
AllRuns.append(AllRun)
bestpara.append(best)
rmae.append(rmae_s)
rmse.append(rmse_s)
rscore.append(rscore_s)
#call the function AutoLGBMR
AllRun,best,pred,rmae_s,rmse_s,rscore_s=AutoLGBMR(x_train, x_test, y_train, y_test)
AllRuns.append(AllRun)
bestpara.append(best)
rmae.append(rmae_s)
rmse.append(rmse_s)
rscore.append(rscore_s)

Auto_Result=pd.concat([pd.DataFrame(rscore, columns=['RScore'], index=['RFR','XGBR','LGBMR']),
                       pd.DataFrame(rmse, columns=['RMSE'], index=['RFR','XGBR','LGBMR']),
                       pd.DataFrame(rmae, columns=['RMAE'], index=['RFR','XGBR','LGBMR']),
                       pd.DataFrame(bestpara, index=['RFR','XGBR','LGBMR']),
                       pd.DataFrame(AllRuns, columns=['AllRuns'], index=['RFR','XGBR','LGBMR'])
                       ],axis=1,sort=False)
   
Run_Model=mode([Auto_Result.RScore.idxmax(),Auto_Result.RMSE.idxmin(),Auto_Result.RMAE.idxmin()])
#-------------------------------------------END-------------------------------------------------#
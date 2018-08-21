# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:14:44 2018

@author: Abhishek S
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.genmod.families import Poisson
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import normalize




os.chdir("E:/Analytics Vidya/ScoreData")
train=pd.read_csv("TV_Data1.csv")
train.head()
train.set_index('household_id',inplace=True)
train.info()
train.describe()
target="NXT_DURATION"
validation=pd.read_csv("TV_Data2.csv")
validation.set_index('household_id',inplace=True)
validation.shape

#without imputing
train_c=train.copy()
train_c.dropna(how='any',axis=0,inplace=True)
train_c.shape
y=train_c[target]
del train_c[target]

ssc=StandardScaler()
ssc.fit(train_c)
x=ssc.transform(train_c)

regr=LinearRegression()
regr.fit(x,y)
regr.score(x,y)
regr.coef_
y_pred=regr.predict(x)
r2_score(y,y_pred) #training  set r_square

y_val=validation[target]
dict_to_fill=dict(zip(train_c.iloc[:,0:14].columns,np.nanmedian(train_c.iloc[:,0:14],axis=0)))
validation.fillna(value=dict_to_fill,inplace=True)
validation.info()
validation=validation.loc[:,list(train_c.columns)]
validation.shape
x_val=ssc.transform(validation)
y_pred_wo_imputed=regr.predict(x_val)
r2_score(y_val,y_pred_wo_imputed)#r_square for  validation using without imputing missing values



#with imputation Model
train_i=train.copy()
train_i.iloc[:,0:14].describe()
col_to_drop=["st_wkend_day_viewer","st_wkday_watcher","cnt_channel_changed_lstwk","st_tot_view_lst_l3m","st_tot_view_lst_mth","time_SAB_view","time_Genre_cartoon",\
             "time_star_view","time_hindi_view","wkday_watcher","st_wkday_watcher_star","avg_viewership_dt_title","cnt_channel_changed_lst_mth",\
             "st_tot_view_lst_month","time_eng_view","time_Tamil_view",'time_Genre_film','time_Genre_news',\
              'time_Genre_cricket','time_Genre_action','cnt_dist_channel_lst_wk_mth','Freq_of_days_view_startv_l3mth','Freq_of_days_view_startv_lst_mth']+['st_cnt_dist_channel_L3M',
 'cnt_dist_channel_lst_wk',
 'time_Genre_Film_Songs',
 'max_time_spent_star_weekend_mth',
 'avg_viewership_dt',
 'time_viewed_lst4wk',
 'st_cnt_dist_channel_lst_wk_mth',
 'avg_viewership_dt_lst_mth',
 'st_cnt_dist_channel_lst_wk',
 'st_wkend_day_viewer_star',
 'time_Telugu_view',
 'wkend_day_viewer',
 'num',
 'max_time_spent_star_weekend',
 'time_Genre_others','Freq_of_days_view_tv_lst_mth']

dict_to_fill=dict(zip(train_i.iloc[:,0:14].columns,np.nanmedian(train_i.iloc[:,0:14],axis=0)))
train_i.fillna(value=dict_to_fill,inplace=True)
train_i.drop(col_to_drop,axis=1,inplace=True)
train_i.shape
yi=train_i[target]
del train_i[target]
train_i.shape
ssc_i=StandardScaler()
ssc_i.fit(train_i)
xi=ssc_i.transform(train_i)
regr_i=LinearRegression()
regr_i.fit(xi,yi)
regr_i.score(xi,yi)# r^2 for training data Model 2


#residuals=yi-regr_i.predict(xi)
#y_fitted=regr_i.predict(xi)

#Same modelling just done with statsmodels to get the model table
np.set_printoptions(suppress=True)
xii=sm.add_constant(xi)
model=sm.OLS(yi,xii)
results=model.fit()
results.summary()


np.set_printoptions(suppress=True)
result_table=pd.DataFrame({"columns":["Constant"]+list(train_i.columns),"params":results.params,"pvalues":np.round(results.pvalues,4)})
result_table['abs_params']=result_table['params'].apply(lambda x:abs(x))
result_table.sort_values('abs_params',ascending=False) # Finding important features


#Table to get variance inflation factor
colname=list()
vif=list()
for i in range(xi.shape[1] -1):
    print(train_i.columns[i]," ",variance_inflation_factor(xi,i))
    vif.append(variance_inflation_factor(xi,i))
    colname.append(train_i.columns[i])

vif_table=pd.DataFrame({"colnames":colname,"vif":vif})

#Scoring validation data
validation=pd.read_csv("TV_Data2.csv")
validation.set_index('household_id',inplace=True)
validation.fillna(value=dict_to_fill,inplace=True)
validation.shape
y_val_i=validation[target]
validation=validation.loc[:,train_i.columns]
xi_val=ssc_i.transform(validation)
y_val_w_imp=results.predict(np.concatenate([np.ones(xi_val.shape[0]).reshape(-1,1),xi_val],axis=1)) # Scoring TV data2 dataset
1- np.sum(np.power(y_val_i.values-y_val_w_imp,2))/np.sum(np.power(y_val_i-np.mean(yi),2)) #r_square on validation for Model 2

#Testing for outliers and their removal
testclass=OLSInfluence(results)
hatvalues=testclass.influence
hatvalues[1509]
hat_table=pd.DataFrame({"obs":np.arange(0,xii.shape[0]),"influence":hatvalues})
hat_table.sort_values('influence',ascending=False,inplace=True)
hat_table['values_normalized']=normalize(hat_table['influence'].values.reshape(-1,1),norm='l2',axis=0)
outliers=hat_table.head(8).obs.values

#Retrain the model after removing the outliers
xii_outlier=sm.add_constant(xi)
xii_outlier=np.delete(xii_outlier,outliers,axis=0)
yi_outlier=np.delete(yi.values,outliers)
model_outliers=sm.OLS(yi_outlier,xii_outlier)
results_outliers=model_outliers.fit()
results_outliers.summary() # r^2 for training data Model 3

y_val_w_imp_outlier=results_outliers.predict(np.concatenate([np.ones(xi_val.shape[0]).reshape(-1,1),xi_val],axis=1)) # Scoring TV data2 dataset after removing outliers
1- np.sum(np.power(y_val_i.values-y_val_w_imp_outlier,2))/np.sum(np.power(y_val_i-np.mean(yi),2)) #r_square on validation for Model 3

result_table_outlier=pd.DataFrame({"columns":["Constant"]+list(train_i.columns),"params":results_outliers.params,"pvalues":np.round(results_outliers.pvalues,4)})
result_table_outlier['abs_params']=result_table_outlier['params'].apply(lambda x:abs(x))
result_table_outlier.sort_values('abs_params',ascending=False) # Finding important features




# Plotting predicted vs actual
plt.scatter(y_val_i,y_val_w_imp_outlier)
plt.title("Validation set with Imputed missing value and outliers")
plt.xlabel("Y actual")
plt.ylabel("Y Predicted")
plt.show()


# Predicting the ratings for the 4th week.

total_viewership_pred=np.sum(y_val_w_imp_outlier)
total_viewership_actual=np.sum(y_val_i)
rating_predicted=total_viewership_pred*100/(7*30*validation.shape[0])
rating_actual=total_viewership_actual*100/(7*30*validation.shape[0])

accuracy=(rating_predicted/rating_actual -1)*100












train.loc[~train.st_time_viewed_lst4wk.isnull(),"NXT_DURATION"].mean()
train.info()
yi=train[target]
yi.index=train.household_id
yi_imputed=yi.loc[train.st_time_viewed_lst4wk.isnull()]

train.set_index('household_id',inplace=True)
train_st1=train.dropna(how='any')
train_st1.info()
target="NXT_DURATION"

train_i=train_st1.copy()
train_i.iloc[:,0:14].describe()
col_to_drop=["st_wkend_day_viewer","st_wkday_watcher","cnt_channel_changed_lstwk","st_tot_view_lst_l3m","st_tot_view_lst_mth","time_SAB_view","time_Genre_cartoon",\
             "time_star_view","time_hindi_view","wkday_watcher","st_wkday_watcher_star","avg_viewership_dt_title","cnt_channel_changed_lst_mth",\
             "st_tot_view_lst_month","time_eng_view","time_Tamil_view",'time_Genre_film','time_Genre_news',\
              'time_Genre_cricket','time_Genre_action','cnt_dist_channel_lst_wk_mth','Freq_of_days_view_startv_l3mth','Freq_of_days_view_startv_lst_mth']+['st_cnt_dist_channel_L3M',
 'cnt_dist_channel_lst_wk',
 'time_Genre_Film_Songs',
 'max_time_spent_star_weekend_mth',
 'avg_viewership_dt',
 'time_viewed_lst4wk',
 'st_cnt_dist_channel_lst_wk_mth',
 'avg_viewership_dt_lst_mth',
 'st_cnt_dist_channel_lst_wk',
 'st_wkend_day_viewer_star',
 'time_Telugu_view',
 'wkend_day_viewer',
 'num',
 'max_time_spent_star_weekend',
 'time_Genre_others','Freq_of_days_view_tv_lst_mth']

dict_to_fill=dict(zip(train_i.iloc[:,0:14].columns,np.nanmedian(train_i.iloc[:,0:14],axis=0)))
train_i.fillna(value=dict_to_fill,inplace=True)
train_i.drop(col_to_drop,axis=1,inplace=True)
train_i.shape
yi=train_i[target]
del train_i[target]
train_i.shape
ssc_i=StandardScaler()
ssc_i.fit(train_i)
xi=ssc_i.transform(train_i)
regr_i=LinearRegression()
regr_i.fit(xi,yi)
regr_i.score(xi,yi)
yi_pred=regr_i.predict(xi)
yi_pred=pd.DataFrame({"yi_pred":yi_pred})
yi_pred.index=train_i.index
yi_pred.shape
n_train=train.merge(yi_pred,how='left',left_index=True,right_index=True)
n_train.info()
n_train.loc[n_train.st_time_viewed_lst4wk.isnull(),"yi_pred"]=0
r2_score(n_train.NXT_DURATION,n_train.yi_pred) #r_square on validation for Model 4


validation=pd.read_csv("TV_Data2.csv")
validation.set_index('household_id',inplace=True)
validation.shape
validation.head()
validation_target=validation[[target,"st_time_viewed_lst4wk"]]
validation=validation.loc[~validation.st_time_viewed_lst4wk.isnull(),:]
y_val_i=validation[target]
validation=validation.loc[:,train_i.columns]
xi_val=ssc_i.transform(validation)
y_val_w_imp=regr_i.predict(xi_val) # Scoring TV data2 dataset
y_val_w_imp.shape
y_val_w_imp=pd.DataFrame({"y_val_w_imp":y_val_w_imp})
y_val_w_imp.index=validation.index
n_validation=validation_target.merge(y_val_w_imp,how='left',left_index=True,right_index=True)
n_validation.info()
n_validation.loc[n_validation.st_time_viewed_lst4wk.isnull(),"y_val_w_imp"]=0

1- np.sum(np.power(n_validation[target].values-n_validation.y_val_w_imp,2))/np.sum(np.power(n_validation[target]-np.mean(train[target]),2)) #r_square on validation for Model 4


validation.loc[validation.st_time_viewed_lst4wk.isnull(),"NXT_DURATION"].mean()

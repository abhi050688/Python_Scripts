# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:58:52 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA


os.chdir('E:/Analytics Vidya/AbInBev/train')


i_volume=pd.read_csv('industry_volume.csv')
i_volume.head()
np.datetime64(i_volume.YearMonth,'%Y%M')
dt.datetime.strptime(str(i_volume.YearMonth[0]),'%Y%m')
pd.datetime.strptime(str(i_volume.YearMonth[0]),'%Y%m')
dateparser=lambda x:pd.datetime.strptime(x,'%Y%m')
i_volume=pd.read_csv('industry_volume.csv',index_col='YearMonth',parse_dates=True,date_parser=dateparser)
i_volume['2013']
i_volume.head()
original_data=i_volume
i_volume=original_data['2014':]
orig=plt.plot(i_volume,color='blue',label='Original')
mn=plt.plot(pd.rolling_mean(i_volume,window=12),color='red',label='Rolling_mean')
vr=plt.plot(pd.rolling_std(i_volume,window=12),color='green',label='Rolling_var')
plt.legend(loc='best')
plt.show(block=False)
original_data.head()
dftest=adfuller(original_data.Industry_Volume,autolag='AIC')
dftest=adfuller(np.log(original_data.Industry_Volume),autolag='AIC')

def dft(df):
    dftest=adfuller(df.Industry_Volume,autolag='AIC')
    print 'ADF statistics %f'% dftest[0]
    print 'p-values %f'% dftest[1]
    print 'critical values'
    for keys,values in dftest[4].items():
        print('\t%s = %f'% (keys,values))


orig_log=np.log(original_data)
plt.plot(pd.rolling_mean(orig_log,window=12))

dft((orig_log-pd.rolling_mean(orig_log,window=12)).dropna())


orig_log.rolling(window=12).mean()
expweighted_avg=orig_log.ewm(halflife=12).mean()
dft(orig_log-expweighted_avg)
lag_acf=acf(orig_log,nlags=20)
lag_pacf=pacf(orig_log,nlags=20,method='ols')

plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(orig_log)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(orig_log)),linestyle='--',color='grey')
plt.tight_layout()
for i,j in enumerate(lag_acf):
    print i,' ',j

q=6
p=2


plt.plot(lag_pacf)
lag_pacf
1.96/

dft((orig_log-orig_log.shift()).dropna())
d=1

q=5 <-res
#orig_log=orig_log['2014':]
orig_log_diff=orig_log-orig_log.shift()

for i in xrange(10):
    model=ARIMA(orig_log,order=(2,1,4))
    results_AR=model.fit(disp=-1)
    plt.plot(orig_log_diff)
    plt.plot(results_AR.fittedvalues,color='red')
    plt.title('RSS: %f'% ((sum((results_AR.fittedvalues-orig_log_diff.Industry_Volume[1:])**2))))
    r2=1- sum((results_AR.fittedvalues-orig_log_diff.Industry_Volume[1:])**2)/sum((orig_log_diff.Industry_Volume[1:]-np.mean(orig_log_diff.Industry_Volume[1:]))**2)
    print r2

pred_arima_diff=pd.Series(results_AR.fittedvalues)
pred_arima_diff.head()
pred_diff_cumsum=pred_arima_diff.cumsum()
pred_diff_cumsum.head()
pred_arima_log=pd.Series(0.,index=orig_log.index)
pred_arima_log.head()
pred_arima_log.iloc[:]=orig_log.iloc[0,0]
pred_arima_log=pred_arima_log.add(pred_diff_cumsum,fill_value=0)
pred_arima_log=np.exp(pred_arima_log)
plt.plot(original_data)
plt.plot(pred_arima_log,color='red')

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:16:03 2018

@author: Abhishek S
"""

import os
import pandas as pd
import numpy as np
import datetime as dt

data=pd.read_csv('E:/Analytics Vidya/Khan Academy/candidate_test_funnel_data.csv',parse_dates=[0])
data.sort_values('timestamp',inplace=True)

?pd.to_datetime
dt.datetime.strptime?
pd.to_datetime(data.timestamp,format="%Y-%m-%d %H:%M:%S")
data.set_index('timestamp',inplace=True)
data['product'].unique()
data['product']=data['product'].astype('category')

data.head()
data.info()
data.mission.unique()
data_pageview=data.loc[data.conversion=='pageview',['product','domain','subject','topic','tutorial','mission']]
data_pageview.info()
total_visitis=data.resample('D').groupby(['timestamp'])['user_id','session_id'].nunique()
data.resample('D')['user_id','session_id'].nunique()
data.user_id=data.user_id.astype('str')
data.session_id=data.session_id.astype('str')
data['visit']=data.user_id+data.session_id
visits=pd.DataFrame(data.resample('D')['visit'].nunique())
data.resample('D').groupby('conversion')['visit'].nunique()
conv=data.groupby([pd.Grouper(freq='D'),'conversion'])['visit'].agg({'tv':pd.Series.nunique})
prop=visits.join(conv)
prop['usage%']=round(prop.tv*100/prop.visit,2)
data['new_conversion']=data.conversion
data.loc[data.new_conversion.isin(['hompage_view','login','pageview']),'new_conversion']='pageview'
data['conversion']=data['new_conversion']
data.loc[data.conversion=='homepage_view','URI'].unique()
data['new_URI']=data['URI'].str.replace('"','').str.split('/')
a=data[['new_URI']][0:7]
a.fillna(value={'new_URI':['']},inplace=True)
a=pd.DataFrame(a.new_URI.values.tolist())

data.head()
data_sub=data[['timestamp','URI','conversion','visit']]
flow=data.sort_values(by=['visit','timestamp'])
col_to_dup=list(data.columns)
col_to_dup.remove('timestamp')
flow=flow.drop_duplicates(subset=col_to_dup)
pd.DataFrame.drop_duplicates(*)
data.URI.unique()[:20]
data_sub.head()
data_sub['URI']=data_sub['URI'].str.replace('"','')
data_sub['page_break']=data_sub['URI'].str.split('/')
pages=data_sub['page_break'].apply(pd.Series)
pages.head()
pages[pages=='']=np.nan
pages.info()
pages.dropna(how='all',inplace=True,axis=1)
pages.shape
pages.columns=['page'+str(i) for i in range(pages.shape[1])]
pages.page0.unique()
data_sub=pd.concat([data_sub,pages],axis=1)
data_sub.head()
fp=data_sub.groupby('page0')['visit'].nunique().sort_values(ascending=False)
fp=fp[fp>50]
fp_index=fp.index.tolist()
data_sub_p=data_sub.loc[data_sub.page0.isin(fp_index),'visit'].drop_duplicates()
data_sub_p.shape
data_sub_pvisit=data_sub[data_sub.visit.isin(data_sub_p)]
data_sub_pvisit.head(20)
#data_sub_pvisit=data_sub_pvisit.loc[:,data_sub_pvisit.columns.str.startswith('page')].head()
data_sub_pvisit['page_no']=np.sum(~data_sub_pvisit[['page0','page1','page2','page3','page4','page5']].isna(),axis=1)
data_sub_pvisit.head()
data_sub_pvisit.groupby('page4')['page4'].count().sort_values(ascending=False)
data_sub_pvisit.sort_values(['visit','timestamp'],inplace=True)
frst=data_sub_pvisit.groupby('visit')['page_no'].first()
frst.head()


# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:04:19 2017

@author: evan
"""

import pandas as pd
import numpy as np

#读取原数据
shop_info = pd.read_csv(r'C:\Users\evan\Desktop\tc\训练数据-ccf_first_round_shop_info.csv')
user_info = pd.read_csv(r'C:\Users\evan\Desktop\tc\训练数据-ccf_first_round_user_shop_behavior.csv')
user_info_ab = pd.read_csv(r'C:\Users\evan\Desktop\tc\AB榜测试集-evaluation_public.csv')

#训练数据join
user_shop_info = user_info.merge(shop_info, left_on='shop_id',right_on='shop_id')

#处理wifi_infos
def wifi_info_split(wifi_info):
    return [wifi.split('|') for wifi in wifi_info.split(';')]
    
user_shop_info['wifi_infos_split'] = user_shop_info['wifi_infos'].apply(wifi_info_split)
  
#取某一商铺
mall_id = 'm_690'

mall_user_shop_info = user_shop_info.loc[user_shop_info.loc[:,'mall_id']==mall_id,:]
  
#wifi强度one-hot数据
def wifi_str_onehot(mall_df):
    user_wifi_dict = {}
    for index,row in mall_df.iterrows():
        user_wifi_dict[index] ={}
        for i in row['wifi_infos_split']:
            if i[0] in user_wifi_dict[index].keys():
                user_wifi_dict[index][i[0]] = (int(i[1])+user_wifi_dict[index][i[0]])/2
            else:
                user_wifi_dict[index][i[0]] = int(i[1])
    return pd.DataFrame(user_wifi_dict).T
    
user_wifi_str_onehot = wifi_str_onehot(mall_user_shop_info)
user_wifi_str_onehot['shop_id'] = mall_user_shop_info['shop_id']
user_wifi_str_onehot_fillna = user_wifi_str_onehot.fillna(-999)
X_cols = [col for col in user_wifi_str_onehot_fillna.columns if col!='shop_id']

#分训练集和测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(\
     user_wifi_str_onehot_fillna[X_cols],user_wifi_str_onehot_fillna['shop_id'] , test_size=0.3, random_state=42)

X_train_df = pd.DataFrame(X_train,columns=X_cols)
X_train_df['shop_id'] = y_train 
X_test_df = pd.DataFrame(X_test,columns=X_cols)
X_test_df['shop_id'] = y_test


#建一个无监督的KNN
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X_train_df[X_cols])



#计算店铺的记录数，并晒出记录数过少的店铺
shop_cnt = user_wifi_str_onehot['shop_id'].value_counts()
shop_filter = shop_cnt[shop_cnt>10].index

#用来存储各个模型的结果
X_test_predict = pd.DataFrame()

#取某店铺的样本
for shop_id in shop_filter:
    print (shop_id)
    #shop_id = 's_684235'
    shop_x = X_train_df.loc[X_train_df.loc[:,'shop_id']==shop_id,:][X_cols]
    
    #取某店铺的周围样本
    n=1
    k=20
    while n<=shop_x.shape[0]*5:
        indices = nbrs.kneighbors(shop_x,n_neighbors =k,return_distance=False)
        indces_list = (indices.reshape((1,shop_x.shape[0]*k))[0].tolist())
        indces_list.extend(shop_x.index.tolist())
        indces_set = set(indces_list)
        n = len(indces_set)
        k+=5
    shop_sample = X_train_df.loc[indces_set,:]
    
    #建模
    shop_sample_dropna = shop_sample.dropna(axis=1,thresh=0.98)
    shop_sample_fillna = shop_sample_dropna.fillna(-999)
    
    shop_sample_fillna['y'] = shop_sample_fillna['shop_id'].apply(lambda x: 1 if x==shop_id else 0)
    
    shop_x_col = [col for col in shop_sample_fillna.columns if col not in ('y','shop_id')]
    
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
    
    rf.fit(shop_sample_fillna[shop_x_col],shop_sample_fillna['y'])
    
    #预测
    x_test = X_test_df[shop_x_col]
    cc = rf.predict_proba(x_test)[:,1]
    X_test_predict[shop_id]=cc
    
c = X_test_predict.idxmax(axis=1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,c)




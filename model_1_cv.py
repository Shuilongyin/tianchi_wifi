# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:06:38 2017

@author: memedai
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:22:10 2017

@author: memedai
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def wifi_split(wifi_info):
    wfif_list = wifi_info.split(';')
    wfif_list_list = [ i.split('|') for i in wfif_list]
    return wfif_list_list

def save_obj(obj,mall_id, name ):
    file = 'E:\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\data\\shop_data\\'+mall_id+'\\pkl_'+ name + '.pkl'
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,mall_id ):
    file = 'E:\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\data\\shop_data\\'+mall_id+'\\pkl_'+ name + '.pkl'
    with open(file, 'rb') as f:
        return pickle.load(f)

shop_info = pd.read_csv(r'E:\evan_mime\Python\Project\用户定位天池大赛_20171011\data\训练数据-ccf_first_round_shop_info.csv')
user_act_info = pd.read_csv(r'E:\evan_mime\Python\Project\用户定位天池大赛_20171011\data\训练数据-ccf_first_round_user_shop_behavior.csv')
user_act_info_ab = pd.read_csv(r'E:\evan_mime\Python\Project\用户定位天池大赛_20171011\data\AB榜测试集-evaluation_public.csv')

mall_list = shop_info['mall_id'].unique() #商场列表
user_act_shop = pd.merge(shop_info,user_act_info,on='shop_id') #训练集merge
user_act_shop['wifi_infos_split'] = user_act_shop['wifi_infos'].apply(wifi_split) #拆分wifi
user_act_info_ab['wifi_infos_split'] = user_act_info_ab['wifi_infos'].apply(wifi_split)
mall_count = user_act_shop['mall_id'].value_counts()

shop_lb = preprocessing.LabelEncoder()
shop_lb.fit(shop_info['shop_id'])

mall_cv_dict = {}
mall_cv_avg_dict = {}

for mall_id in mall_list:
    mall_cv_dict[mall_id] = {}
    mall_cv_avg_dict[mall_id] = []
    #生成商场对应的文件夹
    mall_file = 'E:\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\data\\shop_data\\'+mall_id
    if not os.path.exists(mall_file):
        os.makedirs(mall_file)

        
    #拿对应商场数据
    user_act_m_690 = user_act_shop.loc[user_act_shop.loc[:,'mall_id']==mall_id,:] 
    
    #去掉记录数过少的shop
    shop_cnt = user_act_m_690['shop_id'].value_counts()
    shop = shop_cnt[shop_cnt>5].index
    user_act_m_690['shop_in'] = user_act_m_690['shop_id'].apply(lambda x:x in shop)
    user_act_m_690 = user_act_m_690[user_act_m_690['shop_in']]
    
    #去掉次数过少的wifi
    m_690_bssid_num ={}
    for idx in user_act_m_690.index:
        user_wifi = user_act_m_690.loc[idx,'wifi_infos_split']
        for i in user_wifi:
            if i[0] not in m_690_bssid_num.keys():
                m_690_bssid_num[i[0]] = 1
            else:
                m_690_bssid_num[i[0]] += 1
    
    m_690_bssid_num_df = pd.DataFrame(pd.Series(m_690_bssid_num),columns=['num'])
            
    
    m_690_bssid_num_df_filter = m_690_bssid_num_df.loc[m_690_bssid_num_df.loc[:,'num']>5,:]
    m_690_bssid_num_list_filter = m_690_bssid_num_df_filter.index.tolist()

    #取wifi数据，展成宽表
    m_690_wifi_dict = {}
    for idx in user_act_m_690.index:
        m_690_wifi_dict[idx] = {}
        user_wifi = user_act_m_690.loc[idx,'wifi_infos_split']
        for i in user_wifi:
            if i[0] in m_690_bssid_num_list_filter:
                if i[0] in m_690_wifi_dict[idx]:
                    m_690_wifi_dict[idx][i[0]] =  (m_690_wifi_dict[idx][i[0]]+int(i[1]))/2 #同一记录有重复wifi
                else:
                    m_690_wifi_dict[idx][i[0]] = int(i[1])    

    m_690_wifi_df = pd.DataFrame(m_690_wifi_dict) #转成宽表
    
    m_690_wifi_data = pd.DataFrame(m_690_wifi_df.T.values,columns=m_690_wifi_df.index,index=user_act_m_690.index)\
                .astype(np.float64) #处理列和行的索引
    
    m_690_wifi_data_fillna = m_690_wifi_data.fillna(-999) #填缺失值 
    
    #训练随机森林模型1
    X = m_690_wifi_data_fillna
    y = shop_lb.transform(user_act_m_690['shop_id'])
    
    
    
    #生产每个shop的wifi集
    shops_wifi_cnt_dict = {}
    shops_wifi_strsum_dict = {}
    for shop in user_act_m_690['shop_id'].unique():
        shops_wifi_cnt_dict[shop] = {}
        shops_wifi_strsum_dict[shop] = {}
        shop_data = user_act_m_690.loc[user_act_m_690.loc[:,'shop_id']==shop,['shop_id','wifi_infos_split']]
        for idx in shop_data.index:
            for wi in shop_data.loc[idx,'wifi_infos_split']:
                if wi[0] not in shops_wifi_cnt_dict[shop].keys():
                    shops_wifi_cnt_dict[shop][wi[0]] = 1
                    shops_wifi_strsum_dict[shop][wi[0]] = int(wi[1])
                else:
                    shops_wifi_cnt_dict[shop][wi[0]] += 1
                    shops_wifi_strsum_dict[shop][wi[0]] += int(wi[1])
    shops_wifi_stravg_dict = {}
    for shop in user_act_m_690['shop_id'].unique():
        shops_wifi_stravg_dict[shop] = {}
        for wi in shops_wifi_cnt_dict[shop].keys():
            shops_wifi_stravg_dict[shop][wi] = shops_wifi_strsum_dict[shop][wi]/shops_wifi_cnt_dict[shop][wi]
            
    m_690_user_wifi_stravg = {}
    for idx in user_act_m_690.index:
        m_690_user_wifi_stravg[idx] = {}
        for shop in shops_wifi_stravg_dict.keys():
            m_690_user_wifi_stravg[idx][shop+'_wi_stravg'] = 0
            for wi in user_act_m_690.loc[idx,'wifi_infos_split']:
                m_690_user_wifi_stravg[idx][shop+'_wi_stravg']+=(150+shops_wifi_stravg_dict[shop].get(wi[0],-150))
    
    m_690_user_wifi_stravg_df = pd.DataFrame(m_690_user_wifi_stravg).T 
    
    X2 = m_690_user_wifi_stravg_df
    
    #rf_4 = RandomForestClassifier(100,n_jobs=3)
    mall_cv_dict[mall_id]['rf_1_acc'] = []
    mall_cv_dict[mall_id]['rf_4_acc'] = []
    mall_cv_dict[mall_id]['rf_merge_acc'] = []
    
    kf = KFold(n_splits=5, shuffle=True,random_state=None)
    for train_index, test_index in kf.split(np.array(X)):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        X2_train, X2_test = np.array(X2)[train_index], np.array(X2)[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_col = sorted(set(y_train.tolist()))
        
        rf_1 = RandomForestClassifier(100,n_jobs=3)
        rf_1.fit(X_train,y_train)
        
        rf_4 = RandomForestClassifier(100,n_jobs=3)
        rf_4.fit(X2_train,y_train)
        
        rf_1_df = pd.DataFrame(rf_1.predict_proba(X_test),columns=y_col)
        rf_1_acc = accuracy_score(rf_1_df.idxmax(axis=1),y_test)
        mall_cv_dict[mall_id]['rf_1_acc'].append(rf_1_acc)
        
        rf_4_df = pd.DataFrame(rf_4.predict_proba(X2_test),columns=y_col)
        rf_4_acc = accuracy_score(rf_4_df.idxmax(axis=1),y_test)
        mall_cv_dict[mall_id]['rf_4_acc'].append(rf_4_acc)
        
        rf_merge_df = rf_1_df*0.8+rf_4_df*0.2
        rf_merge_acc = accuracy_score(rf_merge_df.idxmax(axis=1),y_test)
        mall_cv_dict[mall_id]['rf_merge_acc'].append(rf_merge_acc)
    mall_avg_list = (pd.DataFrame(mall_cv_dict[mall_id]).sum()/5).tolist()
    mall_cv_avg_dict[mall_id].extend(mall_avg_list)
    
malls_cv_df = pd.DataFrame(mall_cv_avg_dict)
malls_cv_df.to_csv(r'')
with open(r'', 'wb') as f:
    pickle.dump(mall_cv_dict, f, pickle.HIGHEST_PROTOCOL)






















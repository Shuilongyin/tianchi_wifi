# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:12:00 2017

@author: evan
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import pickle
import os

#读取原数据
shop_info = pd.read_csv(r'C:\xww\evan_mime\Python\Project\用户定位天池大赛_20171011\data\训练数据-ccf_first_round_shop_info.csv')
user_info = pd.read_csv(r'C:\xww\evan_mime\Python\Project\用户定位天池大赛_20171011\data\训练数据-ccf_first_round_user_shop_behavior.csv')
user_act_info_ab = pd.read_csv(r'C:\xww\evan_mime\Python\Project\用户定位天池大赛_20171011\data\AB榜测试集-evaluation_public.csv')

def save_obj(obj, name ): #pkl文件保存
    file = 'C:\\xww\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\pkl文件\\mall_wait_shop_20171114\\pkl_'+ name + '.pkl'
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name): #pkl文件load
    file = 'C:\\xww\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\pkl文件\\mall_wait_shop_20171114\\pkl_'+ name + '.pkl'
    with open(file, 'rb') as f:
        return pickle.load(f)

#训练数据join
user_shop_info = user_info.merge(shop_info, left_on='shop_id',right_on='shop_id')

def wifi_split(wifi_info): #拆分wifi
    wfif_list = wifi_info.split(';')
    wfif_list_list = [ i.split('|') for i in wfif_list]
    return wfif_list_list

user_shop_info['wifi_infos_split'] = user_shop_info['wifi_infos'].apply(wifi_split) #训练集拆分wifi
user_act_info_ab['wifi_infos_split'] = user_act_info_ab['wifi_infos'].apply(wifi_split) #ab集拆分wifi

#筛选记录数满足条件的商铺
def filter_shop(mall_df,th=5):
    shop_cnt = mall_df['shop_id'].value_counts()
    shop = shop_cnt[shop_cnt>th].index
    mall_df['shop_in'] = mall_df['shop_id'].apply(lambda x:x in shop)
    mall_df = mall_df[mall_df['shop_in']]
    return mall_df, shop_cnt


#计算bssid在某商场出现的数量,以及连接的数量
def calc_bssid(mall_df):
    m_690_bssid_num = {}
    m_690_bssid_connect_num = {}
    for idx in mall_df.index:
        user_wifi = mall_df.loc[idx,'wifi_infos_split']
        for i in user_wifi:
            if i[0] not in m_690_bssid_num.keys():
                m_690_bssid_num[i[0]] = 1
            else:
                m_690_bssid_num[i[0]] += 1
        for i in user_wifi:
            if i[0] not in m_690_bssid_connect_num.keys() and i[2]=='true':
                m_690_bssid_connect_num[i[0]] = 1
            elif i[0]  in m_690_bssid_connect_num.keys() and i[2]=='true':
                m_690_bssid_connect_num[i[0]] += 1
    return m_690_bssid_num,m_690_bssid_connect_num

#计算bssid在某商场出现的天数
def calc_bssid_day(mall_df):
    mall_df['date'] = mall_df['time_stamp'].apply(lambda x:pd.to_datetime(x).date())
    m_690_bssid_day = {}
    for idx in mall_df.index:
        user_wifi = mall_df.loc[idx,'wifi_infos_split']
        for i in user_wifi:
            if i[0] not in m_690_bssid_day.keys():
                m_690_bssid_day[i[0]] = set([mall_df.loc[idx,'date']])
            else:
                m_690_bssid_day[i[0]] = m_690_bssid_day[i[0]].union(set([mall_df.loc[idx,'date']]))
    m_690_bssid_day_num={}            
    for wi in m_690_bssid_day.keys():
        m_690_bssid_day_num[wi] = len(m_690_bssid_day[wi])
    return pd.Series(m_690_bssid_day_num)

#产出每个店铺的wifi强度聚合
def wifi_str_shops_collect(mall_df):
    shops_wifi_cnt_dict = {}
    shops_wifi_strsum_dict = {}
    for shop in mall_df['shop_id'].unique():
        shops_wifi_cnt_dict[shop] = {}
        shops_wifi_strsum_dict[shop] = {}
        shop_data = mall_df.loc[mall_df.loc[:,'shop_id']==shop,['shop_id','wifi_infos_split']]
        for idx in shop_data.index:
            for wi in shop_data.loc[idx,'wifi_infos_split']:
                if wi[0] not in shops_wifi_cnt_dict[shop].keys():
                    shops_wifi_cnt_dict[shop][wi[0]] = 1
                    shops_wifi_strsum_dict[shop][wi[0]] = int(wi[1])
                else:
                    shops_wifi_cnt_dict[shop][wi[0]] += 1
                    shops_wifi_strsum_dict[shop][wi[0]] += int(wi[1])
    shops_wifi_stravg_dict = {}
    for shop in mall_df['shop_id'].unique():
        shops_wifi_stravg_dict[shop] = {}
        for wi in shops_wifi_cnt_dict[shop].keys():
            shops_wifi_stravg_dict[shop][wi] = shops_wifi_strsum_dict[shop][wi]/shops_wifi_cnt_dict[shop][wi]
    return shops_wifi_cnt_dict,shops_wifi_stravg_dict

#产出每个店铺的wifi 出现次数的tf-idf聚合
def wifi_cnt_tfidf_shops_collect(mall_df,m_690_bssid_num,shops_wifi_cnt_dict,shop_cnt):
    shops_wifi_tfidf_dict={}
    for shop in mall_df['shop_id'].unique():
        shops_wifi_tfidf_dict[shop] = {}
        for wi in shops_wifi_cnt_dict[shop].keys():
            shops_wifi_tfidf_dict[shop][wi] = (shops_wifi_cnt_dict[shop][wi]/shop_cnt[shop])/\
               (m_690_bssid_num[wi]/mall_df.shape[0]) #tf-idf
    return shops_wifi_tfidf_dict
    
#产出每个人针对每个店铺的wifi强度聚合的变量
def wifi_str_shops_collect_data(mall_df,shops_wifi_stravg_dict):
    m_690_user_wifi_stravg = {}
    for idx in mall_df.index:
        m_690_user_wifi_stravg[idx] = {}
        for shop in shops_wifi_stravg_dict.keys():
            m_690_user_wifi_stravg[idx][shop+'_wi_stravg'] = 0
            for wi in mall_df.loc[idx,'wifi_infos_split']:
                m_690_user_wifi_stravg[idx][shop+'_wi_stravg']+=(150+shops_wifi_stravg_dict[shop].get(wi[0],-150))
    
    m_690_user_wifi_stravg_df = pd.DataFrame(m_690_user_wifi_stravg).T
    return m_690_user_wifi_stravg_df

#产出每个人针对每个店铺的wifi出现人数的tf-idf聚合的变量
def wifi_cnt_tfidf_shops_collect_data(mall_df, shops_wifi_tfidf_dict):
    m_690_user_wifi_tfidf = {}
    for idx in mall_df.index:
        m_690_user_wifi_tfidf[idx] = {}
        for shop in shops_wifi_tfidf_dict.keys():
            m_690_user_wifi_tfidf[idx][shop+'_wi_cnt'] = 0
            for wi in mall_df.loc[idx,'wifi_infos_split']:
                m_690_user_wifi_tfidf[idx][shop+'_wi_cnt']+=shops_wifi_tfidf_dict[shop].get(wi[0],0)
    m_690_user_wifi_tfidf_df = pd.DataFrame(m_690_user_wifi_tfidf).T 
    return m_690_user_wifi_tfidf_df

mall_list =  set(user_shop_info['mall_id'].tolist() )
mall_recall = {}



for mall_id in mall_list:
#取某一商铺
    print (mall_id)
    mall_id_wait = {}
    mall_id_wait_ab = {}
    
    user_act_m_690 = user_shop_info.loc[user_shop_info.loc[:,'mall_id']==mall_id,:]
    
    #过滤掉某些数量过少的店铺
    user_act_m_690, shop_cnt = filter_shop(user_act_m_690,th=2)
    
    #过滤wifi，为后一步做准备
    m_690_bssid_num,m_690_bssid_connect_num = calc_bssid(user_act_m_690)
    m_690_bssid_day_num = calc_bssid_day(user_act_m_690)
    bssid_filter = set(m_690_bssid_day_num[m_690_bssid_day_num>3].index)
    
    #根据bssid_filter清洗原数据
    user_act_m_690['wifi_infos_split'] = user_act_m_690['wifi_infos_split'].apply(lambda x:[k for k in x if k[0] in bssid_filter])
    
    shops_wifi_cnt_dict,shops_wifi_stravg_dict = wifi_str_shops_collect(user_act_m_690)
#    shops_wifi_tfidf_dict = wifi_cnt_tfidf_shops_collect(user_act_m_690,m_690_bssid_num,shops_wifi_cnt_dict,shop_cnt)
    
    m_690_user_wifi_stravg_df_train = wifi_str_shops_collect_data(user_act_m_690,shops_wifi_stravg_dict)
#    m_690_user_wifi_tfidf_df_train = wifi_cnt_tfidf_shops_collect_data(user_act_m_690, shops_wifi_tfidf_dict)
    
    
#    m_690_concen_train = m_690_user_wifi_stravg_df_train.merge(m_690_user_wifi_tfidf_df_train,left_index=True,right_index=True\
#                                                     ,how='outer')
    
    shop_lb = preprocessing.LabelEncoder() 
    
    X = m_690_user_wifi_stravg_df_train
    y = pd.Series(shop_lb.fit_transform(user_act_m_690['shop_id']),index=user_act_m_690.index)
    
    knn = KNeighborsClassifier(n_neighbors=10,n_jobs =-1)
    knn.fit(X, y) 
    
    y_col = sorted(set(y.tolist()))
    
    knn_df = pd.DataFrame(knn.predict_proba(m_690_user_wifi_stravg_df_train),columns=y_col,index=user_act_m_690.index)
    
    
    knn_df_rank = knn_df.rank(axis=1,ascending=False,method ='max')
    
    top=10
    cover_num=0
    for idx in knn_df.index:
        wait_list = knn_df_rank.loc[idx,knn_df_rank.loc[idx,:]<=top].index.tolist()
        mall_id_wait[idx] = shop_lb.inverse_transform(wait_list).tolist()
        if y[idx] in wait_list:
            cover_num+=1
    recall_ratio =cover_num/knn_df.shape[0]
    print ('recall_ratio:',recall_ratio)         
    mall_recall[mall_id] =  recall_ratio


#ab集
    user_act_m_690_ab = user_act_info_ab.loc[user_act_info_ab.loc[:,'mall_id']==mall_id,:] 
    user_act_m_690_ab['wifi_infos_split'] = user_act_m_690_ab['wifi_infos_split'].apply(lambda x:[k for k in x if k[0] in bssid_filter])   

    m_690_user_wifi_stravg_df_ab = wifi_str_shops_collect_data(user_act_m_690_ab,shops_wifi_stravg_dict)
#    m_690_user_wifi_tfidf_df_ab = wifi_cnt_tfidf_shops_collect_data(user_act_m_690_ab, shops_wifi_tfidf_dict)

#    m_690_concen_ab = m_690_user_wifi_stravg_df_ab.merge(m_690_user_wifi_tfidf_df_ab,left_index=True,right_index=True\
#                                                     ,how='outer')
    knn_df_ab = pd.DataFrame(knn.predict_proba(m_690_user_wifi_stravg_df_ab),columns=y_col,index=user_act_m_690_ab.row_id)
    knn_df_rank_ab = knn_df_ab.rank(axis=1,ascending=False,method ='max')
    
    for idx in knn_df_ab.index:
        wait_list_ab = knn_df_rank_ab.loc[idx,knn_df_rank_ab.loc[idx,:]<=top].index.tolist()
        mall_id_wait_ab[idx] = shop_lb.inverse_transform(wait_list_ab).tolist()
    
    save_obj(mall_id_wait,mall_id)
    save_obj(mall_id_wait_ab,mall_id+'_ab')
    


save_obj(mall_recall,'mall_recall')






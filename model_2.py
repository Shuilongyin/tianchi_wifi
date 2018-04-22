# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:39:46 2017

@author: memedai
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

def wifi_split(wifi_info): #拆分wifi
    wfif_list = wifi_info.split(';')
    wfif_list_list = [ i.split('|') for i in wfif_list]
    return wfif_list_list

def save_obj(obj, name ): #pkl文件保存
    file = 'C:\\xww\\evan_mime\\Python\Project\\用户定位天池大赛_20171011\\data\\shop_dat\\pkl_'+ name + '.pkl'
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name): #pkl文件load
    file = 'C:\\xww\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\data\\shop_data\\pkl_'+ name + '.pkl'
    with open(file, 'rb') as f:
        return pickle.load(f)
    
shop_info = pd.read_csv(r'C:\xww\evan_mime\Python\Project\用户定位天池大赛_20171011\data\训练数据-ccf_first_round_shop_info.csv')
user_act_info = pd.read_csv(r'C:\xww\evan_mime\Python\Project\用户定位天池大赛_20171011\data\训练数据-ccf_first_round_user_shop_behavior.csv')
user_act_info_ab = pd.read_csv(r'C:\xww\evan_mime\Python\Project\用户定位天池大赛_20171011\data\AB榜测试集-evaluation_public.csv')

mall_list = shop_info['mall_id'].unique() #商场列表
user_act_shop = pd.merge(shop_info.loc[:,['shop_id','category_id','price','mall_id']],user_act_info,on='shop_id') #训练集merge，将mall加入
user_act_shop['wifi_infos_split'] = user_act_shop['wifi_infos'].apply(wifi_split) #训练集拆分wifi
user_act_info_ab['wifi_infos_split'] = user_act_info_ab['wifi_infos'].apply(wifi_split) #ab集拆分wifi
mall_count = user_act_shop['mall_id'].value_counts() #统计训练集中mall的记录数

#y one-hot
shop_lb = preprocessing.LabelEncoder()
shop_lb.fit(shop_info['shop_id'])    

#处理经纬度
mail_loaction = {}
for l in list(set(shop_info.mall_id)):
    a = sorted(shop_info.longitude[shop_info.mall_id==l])
    b = sorted(shop_info.latitude[shop_info.mall_id == l])
    mail_loaction[l] = [[a[0],a[-1],a[1]-a[0],a[-1]-a[-2]],[b[0],b[-1],b[1]-b[0],b[-1]-b[-2]]]


def lon_map(x):
    a = mail_loaction[x.mall_id][0]
    if x.longitude>=a[0] and x.longitude<=a[1]:
        return x.longitude
    elif x.longitude<a[0]:
        if a[0]-x.longitude<=a[2]:
            return a[0]
        else:
            return (a[0]+a[1])/2
    else:
        if x.longitude-a[1]<=a[3]:
            return a[1]
        else:
            return (a[0]+a[1])/2

def lat_map(x):
    a = mail_loaction[x.mall_id][1]
    if x.latitude>=a[0] and x.latitude<=a[1]:
        return x.latitude
    elif x.latitude<a[0]:
        if a[0]-x.latitude<=a[2]:
            return a[0]
        else:
            return (a[0]+a[1])/2
    else:
        if x.latitude-a[1]<=a[3]:
            return a[1]
        else:
            return (a[0]+a[1])/2

user_act_shop['lon'] = user_act_shop.apply(lambda x:lon_map(x),axis = 1)  
user_act_shop['lat'] = user_act_shop.apply(lambda x:lat_map(x),axis = 1)

user_act_info_ab['lon'] = user_act_info_ab.apply(lambda x:lon_map(x),axis = 1)
user_act_info_ab['lat'] = user_act_info_ab.apply(lambda x:lat_map(x),axis = 1)

#处理时间
user_act_shop['time_stamp'] = user_act_shop['time_stamp'].apply(lambda x:pd.to_datetime(x))
user_act_info_ab['time_stamp'] = user_act_info_ab['time_stamp'].apply(lambda x:pd.to_datetime(x))


user_act_shop['hour'] = user_act_shop['time_stamp'].apply(lambda x:x.hour)
user_act_shop['weekday'] = user_act_shop['time_stamp'].apply(lambda x:x.weekday())

user_act_info_ab['hour'] = user_act_info_ab['time_stamp'].apply(lambda x:x.hour)
user_act_info_ab['weekday'] = user_act_info_ab['time_stamp'].apply(lambda x:x.weekday())

#计算bssid出现的商场数
def calc_bssid_mallnum(user_act):
    bssid_mall= {}
    for idx in user_act.index:
        user_wifi = user_act.loc[idx,'wifi_infos_split']
        for i in user_wifi:
            if i[0] not in bssid_mall.keys():
                bssid_mall[i[0]] = set([user_act.loc[idx,'mall_id']])
            else:
                bssid_mall[i[0]] = bssid_mall[i[0]].union(set([user_act.loc[idx,'mall_id']]))
    bssid_mall_num={}            
    for wi in bssid_mall.keys():
        bssid_mall_num[wi] = len(bssid_mall[wi])
    return pd.Series(bssid_mall_num)

#tt = calc_bssid_mallnum(user_act_shop) #慎用，运行时间很长






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

               
#产出满足条件的wifi强度，one-hot数据
def wifi_str_onehot(mall_df):
    m_690_wifi_dict = {}
    for idx in mall_df.index:
        m_690_wifi_dict[idx] = {}
        user_wifi = mall_df.loc[idx,'wifi_infos_split']
        for i in user_wifi:
            if i[0] in m_690_wifi_dict[idx]:
                m_690_wifi_dict[idx][i[0]] =  (m_690_wifi_dict[idx][i[0]]+int(i[1]))/2 #同一记录有重复wifi
            else:
                m_690_wifi_dict[idx][i[0]] = int(i[1])
    m_690_wifi_df = pd.DataFrame(m_690_wifi_dict) #转成宽表
    
    m_690_wifi_data = pd.DataFrame(pd.DataFrame(m_690_wifi_dict).T.values,columns=m_690_wifi_df.index,\
                                   index=mall_df.index)\
                .astype(np.float64) #处理列和行的索引
    return m_690_wifi_data
    
#产出满足条件的wifi连接，one-hot数据
def wifi_connect_onehot(mall_df,m_690_bssid_connect_num):
    m_690_wifi_connect_dict = {}
    for idx in mall_df.index:
        m_690_wifi_connect_dict[idx] = {}
        user_wifi = mall_df.loc[idx,'wifi_infos_split']
        for i in user_wifi:
            if m_690_bssid_connect_num.get(i[0],0)>=2:
                if i[0] in m_690_wifi_connect_dict[idx]:
                    if i[2]=='true':
                        m_690_wifi_connect_dict[idx][i[0]] =  1 #同一记录有重复wifi
                else:
                    if i[2]=='true':
                        m_690_wifi_connect_dict[idx][i[0]] = 1
                    else:
                        m_690_wifi_connect_dict[idx][i[0]] = np.nan #按缺失存储    
    m_690_wifi_connect_df = pd.DataFrame(m_690_wifi_connect_dict) #转成宽表
    
    m_690_wifi_connect_data = pd.DataFrame(m_690_wifi_connect_df.T.values,columns=m_690_wifi_connect_df.index,index=mall_df.index)\
                .astype(np.float64) #处理列和行的索引
    return m_690_wifi_connect_data

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
               
#产出每个店铺wifi连接数的tf-idf聚合
def wifi_connect_tfidf_shops_collect(mall_df,shop_cnt):
    shops_wifi_connect_dict={}
    for shop in mall_df['shop_id'].unique():
        shops_wifi_connect_dict[shop] = {}
        shop_data = mall_df.loc[mall_df.loc[:,'shop_id']==shop,['shop_id','wifi_infos_split']]
        for idx in shop_data.index:
            for wi in shop_data.loc[idx,'wifi_infos_split']:
                if wi[0] not in shops_wifi_connect_dict[shop].keys() and wi[2]=='true':
                    shops_wifi_connect_dict[shop][wi[0]] = 1
                elif wi[0] in shops_wifi_connect_dict[shop].keys() and wi[2]=='true':
                    shops_wifi_connect_dict[shop][wi[0]] += 1
    
    m_690_bssid_connect_num={}            
    for shop in shops_wifi_connect_dict.keys():
        for bssid in shops_wifi_connect_dict[shop].keys():
            if bssid not in m_690_bssid_connect_num.keys():
               m_690_bssid_connect_num[bssid] =  shops_wifi_connect_dict[shop][bssid]
            else:
               m_690_bssid_connect_num[bssid] +=  shops_wifi_connect_dict[shop][bssid] 
    
    shops_wifi_connect_tfidf_dict={}
    for shop in mall_df['shop_id'].unique():
        shops_wifi_connect_tfidf_dict[shop] = {}
        for wi in shops_wifi_connect_dict[shop].keys():
            shops_wifi_connect_tfidf_dict[shop][wi] = (shops_wifi_connect_dict[shop][wi]/shop_cnt[shop])/\
               (m_690_bssid_connect_num[wi]/mall_df.shape[0]) #tf-idf
    return shops_wifi_connect_tfidf_dict

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

#产出每个人针对每个店铺的wifi连接数的tf-idf聚合的变量
def wifi_connect_tfidf_shops_collect_data(mall_df, shops_wifi_connect_tfidf_dict):
    m_690_user_wifi_connect_tfidf = {}
    for idx in mall_df.index:
        m_690_user_wifi_connect_tfidf[idx] = {}
        for shop in shops_wifi_connect_tfidf_dict.keys():
            m_690_user_wifi_connect_tfidf[idx][shop+'_wi_connect'] = 0
            for wi in mall_df.loc[idx,'wifi_infos_split']:
                m_690_user_wifi_connect_tfidf[idx][shop+'_wi_connect']+=shops_wifi_connect_tfidf_dict[shop].get(wi[0],0)
    m_690_user_wifi_connect_tfidf_df = pd.DataFrame(m_690_user_wifi_connect_tfidf).T 
    return m_690_user_wifi_connect_tfidf_df



malls_info = {} #存储一些mall的记录信息
ab_df = pd.DataFrame()
ab_df_ratio = pd.DataFrame()

############针对每个mall做处理############
#mall_id = 'm_7168'

os.chdir('C:\\xww\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\data')
path = os.getcwd()

i=0
for mall_id in mall_list[10:15]:
    os.makedirs(path + '\\edata\\' + mall_id)
    print (i,mall_id)
    i+=1
    #商场记录信息初始化
    #malls_info[mall_id] = {}
    #malls_info[mall_id]['max_acc'] = 0 #最佳acc
    #malls_info[mall_id]['best_per'] = 0 #最佳融合系数
    
    #取商场的数据
    
    user_act_m_690 = user_act_shop.loc[user_act_shop.loc[:,'mall_id']==mall_id,:] 
    
    user_act_m_690_ab = user_act_info_ab.loc[user_act_info_ab.loc[:,'mall_id']==mall_id,:] 
    
    #过滤掉某些数量过少的店铺
    user_act_m_690, shop_cnt = filter_shop(user_act_m_690,th=2)
    
    #过滤wifi，为后一步做准备
    m_690_bssid_num,m_690_bssid_connect_num = calc_bssid(user_act_m_690)
    m_690_bssid_day_num = calc_bssid_day(user_act_m_690)
    
    
#    m_690_bssid_num_over = set([wi for wi in m_690_bssid_num.keys() if m_690_bssid_num[wi]>6]) #暂不按数量筛选
    bssid_filter = set(m_690_bssid_day_num[m_690_bssid_day_num>3].index)
#    m_690_bssid_jiaoji = (m_690_bssid_num_over.union(m_690_bssid_day_num_over))^(m_690_bssid_num_over^m_690_bssid_day_num_over)
    
    #bssid_filter = [i for i in m_690_bssid_day_num_over if i not in (set(tt[tt>1].index))]
    
    #根据bssid_filter清洗原数据
    user_act_m_690['wifi_infos_split'] = user_act_m_690['wifi_infos_split'].apply(lambda x:[k for k in x if k[0] in bssid_filter])

    user_act_m_690_ab['wifi_infos_split'] = user_act_m_690_ab['wifi_infos_split'].apply(lambda x:[k for k in x if k[0] in bssid_filter])   
    
    #取wifi强度one-hot变量
    m_690_wifi_data = wifi_str_onehot(user_act_m_690)
    m_690_wifi_data_ab = pd.DataFrame(wifi_str_onehot(user_act_m_690_ab),columns=m_690_wifi_data.columns)
    
    m_690_wifi_data_fillna = m_690_wifi_data.fillna(-999)
    m_690_wifi_data_ab_fillna = m_690_wifi_data_ab.fillna(-999)
    
    #取wifi连接one-hot变量
    m_690_wifi_connect_data = wifi_connect_onehot(user_act_m_690,m_690_bssid_connect_num).fillna(0)
    m_690_wifi_connect_data_ab = pd.DataFrame(wifi_connect_onehot(user_act_m_690_ab\
                                                        ,m_690_bssid_connect_num),columns=m_690_wifi_connect_data.columns).fillna(0)
    
    #merge wifi强度和连接one-hot变量
    m_690_wifi_data_merge = m_690_wifi_data_fillna.merge(m_690_wifi_connect_data,left_index=True,right_index=True)
    m_690_wifi_data_merge_ab = m_690_wifi_data_ab_fillna.merge(m_690_wifi_connect_data_ab,left_index=True,right_index=True)
    
    #取y
    y = pd.Series(shop_lb.transform(user_act_m_690['shop_id']),index=user_act_m_690.index)
    
    
    

#k折太慢，退而求其次
    shops_wifi_cnt_dict,shops_wifi_stravg_dict = wifi_str_shops_collect(user_act_m_690)
    shops_wifi_tfidf_dict = wifi_cnt_tfidf_shops_collect(user_act_m_690,m_690_bssid_num,shops_wifi_cnt_dict,shop_cnt)
    shops_wifi_connect_tfidf_dict = wifi_connect_tfidf_shops_collect(user_act_m_690,shop_cnt)

    m_690_user_wifi_stravg_df_train = wifi_str_shops_collect_data(user_act_m_690,shops_wifi_stravg_dict)
    m_690_user_wifi_tfidf_df_train = wifi_cnt_tfidf_shops_collect_data(user_act_m_690, shops_wifi_tfidf_dict)
    m_690_user_wifi_connect_tfidf_df_train = wifi_connect_tfidf_shops_collect_data(user_act_m_690, shops_wifi_connect_tfidf_dict)
 
    m_690_user_wifi_stravg_df_ab = wifi_str_shops_collect_data(user_act_m_690_ab,shops_wifi_stravg_dict)
    m_690_user_wifi_tfidf_df_ab = wifi_cnt_tfidf_shops_collect_data(user_act_m_690_ab, shops_wifi_tfidf_dict)
    m_690_user_wifi_connect_tfidf_df_ab = wifi_connect_tfidf_shops_collect_data(user_act_m_690_ab, shops_wifi_connect_tfidf_dict)

    m_690_concen_train = m_690_user_wifi_stravg_df_train.merge(m_690_user_wifi_tfidf_df_train,left_index=True,right_index=True\
                                                     ,how='outer').merge(m_690_user_wifi_connect_tfidf_df_train,left_index=True,\
                                                     right_index=True,how='outer')
    m_690_concen_ab = m_690_user_wifi_stravg_df_ab.merge(m_690_user_wifi_tfidf_df_ab,left_index=True,right_index=True\
                                                     ,how='outer').merge(m_690_user_wifi_connect_tfidf_df_ab,left_index=True,\
                                                     right_index=True,how='outer')    
    
    m_690_concen_train['hour'] = user_act_m_690['hour']
    m_690_concen_train['weekday'] = user_act_m_690['weekday']
    m_690_concen_train['lat'] = user_act_m_690['lat']
    m_690_concen_train['lon'] = user_act_m_690['lon']
    
    m_690_concen_ab['hour'] = user_act_m_690_ab['hour']
    m_690_concen_ab['weekday'] = user_act_m_690_ab['weekday']
    m_690_concen_ab['lat'] = user_act_m_690_ab['lat']
    m_690_concen_ab['lon'] = user_act_m_690_ab['lon']

    #训练
    rf_1 = RandomForestClassifier(400,n_jobs=-1)
    rf_1.fit(m_690_wifi_data_merge,y)
    
    rf_4 = RandomForestClassifier(400,n_jobs=-1)
    rf_4.fit(m_690_concen_train,y)
    
    y_col = sorted(set(y.tolist()))
    
    rf_1_df_ab = pd.DataFrame(rf_1.predict_proba(m_690_wifi_data_merge_ab),columns=y_col)
    
    rf_4_df_ab = pd.DataFrame(rf_4.predict_proba(m_690_concen_ab),columns=y_col)
    
    rf_df_merge = rf_1_df_ab*0.8+rf_4_df_ab*0.2
    #ab_df_ratio = pd.concat([ab_df_ratio,rf_df_merge])
    
    y_ab = shop_lb.inverse_transform(rf_df_merge.idxmax(axis=1))
    
    user_act_m_690_ab['shop_id'] = y_ab
    user_act_m_690_ab.loc[:,['row_id','shop_id']].to_csv('edata\\'+mall_id+'\\ab.csv',index=False)

    

    


import os
import pickle
import pandas as pd
from datetime import datetime,date
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from functools import reduce
from collections import Counter

from math import cos, sin, atan2, sqrt, pi ,radians, degrees
from sklearn import linear_model

lr = linear_model.LinearRegression()


import xgboost as xgb
os.chdir('C:\\xww\\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\\data')
path = os.getcwd()


df0 = pd.read_csv('训练数据-ccf_first_round_shop_info.csv')
df1 = pd.read_csv('训练数据-ccf_first_round_user_shop_behavior.csv')
df2 = pd.read_csv('AB榜测试集-evaluation_public.csv')

df1 = df1.merge(df0, left_on='shop_id',right_on='shop_id')
df1['row_id'] = df1.index


def if_du1(x):
    if len(x.split(';'))!= len(list(set([l.split('|')[0] for l in x.split(';')]))):
        return 1
    else:
        return 0

def map1(x):
    wifi_info = {}
    for k in [l.split('|') for l in x.split(';')]:
        tf = 0
        if k[2] == 'true':
            tf = 1
        wifi_info[k[0]] = [int(k[1]),tf]
    return wifi_info

def dup(x):
    wifi_info = {}
    for k in [l.split('|') for l in x.split(';')]:
        tf = 0
        if k[2] == 'true':
            tf = 1
        if k[0] not in wifi_info:
            wifi_info[k[0]] = [int(k[1]),tf]
        else:
            #wifi_info[k[0]].append([int(k[1]),tf])
            wifi_info[k[0]][0] = (wifi_info[k[0]][0]+int(k[1]))/2
            wifi_info[k[0]][1] = max(tf,wifi_info[k[0]][1])
    return wifi_info

def wifi_list(x):
    if x.if_du1==0:
        return map1(x.wifi_infos)
    else:
        return dup(x.wifi_infos)

df1['if_du1'] = df1.apply(lambda x:if_du1(x.wifi_infos),axis = 1)
df1['wifi_list'] =  df1.apply(lambda x:wifi_list(x),axis = 1)

df2['if_du1'] = df2.apply(lambda x:if_du1(x.wifi_infos),axis = 1)
df2['wifi_list'] =  df2.apply(lambda x:wifi_list(x),axis = 1)

df1['date'] = df1['time_stamp'].apply(lambda x:pd.to_datetime(x).date())

#计算bssid在某商场出现的天数
def calc_bssid_day(mall_df):
    #mall_df['date'] = mall_df['time_stamp'].apply(lambda x:pd.to_datetime(x).date())
    m_690_bssid_day = {}
    for idx in mall_df.index:
        #print (idx)
        user_wifi = mall_df.loc[idx,'wifi_list']
        for i in user_wifi:
            if i not in m_690_bssid_day.keys():
                m_690_bssid_day[i] = set([mall_df.loc[idx,'date']])
            else:
                m_690_bssid_day[i] = m_690_bssid_day[i].union(set([mall_df.loc[idx,'date']]))
    m_690_bssid_day_num={}            
    for wi in m_690_bssid_day.keys():
        m_690_bssid_day_num[wi] = len(m_690_bssid_day[wi])
    return pd.Series(m_690_bssid_day_num)

    
#m_690_bssid_day_num = calc_bssid_day(df1)
#m_690_bssid_day_num.to_csv(r'm_690_bssid_day_num.csv')
bssid_filter = set(m_690_bssid_day_num[m_690_bssid_day_num>3].index)    

def select_wifi(x):
    b = {}
    for k in x:
        if k in bssid_filter:
            b[k] = x[k]
    return b

df1['wifi_list'] = df1['wifi_list'].apply(lambda x:select_wifi(x))
df2['wifi_list'] = df2['wifi_list'].apply(lambda x:select_wifi(x))


def order_wifi(x):
    b = {}
    c = sorted(x.items(), key=lambda item: item[1][0])
    for i in range(len(x)):
        b[c[i][0]] = i
    return b

df1['wifi_list'] = df1['wifi_list'].apply(lambda x: order_wifi(x))
df2['wifi_list'] = df2['wifi_list'].apply(lambda x: order_wifi(x))

mall_list = list(set(df0.mall_id))

i = 0
for l in mall_list:
    os.makedirs(path + '\\cdata\\' + l)
    print(i, l)
    i += 1
    dff1 = df1[df1['mall_id'] == l][['row_id', 'wifi_list', 'shop_id']]
    dff2 = df2[df2['mall_id'] == l][['row_id', 'wifi_list']]


    order_wifi_dic = {}
    for ind in dff1.index:
        order_wifi_dic[ind] = dff1.loc[ind, 'wifi_list']

    a = pd.DataFrame(order_wifi_dic).T
    a['row_id'] = a.index
    a = a.fillna(-999)

    pre = [k for k in a if k != 'row_id']
    b = pd.merge(a, dff1[['row_id', 'shop_id']], on='row_id')
    b.index = a.index
    ###训练集
    X = b[pre]
    le1 = preprocessing.LabelEncoder()
    le1.fit(list(set(b['shop_id'])))
    Y = le1.transform(b['shop_id'])
    row_id_X = np.array(b['row_id'])

    shop_id_list = list(set(dff1.shop_id))
    a = shop_id_list.copy()
    a.append('row_id')

    train_data_order_wifi = pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=True, random_state=None)

    for train_index, test_index in kf.split(np.array(X)):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        row_id_X_train, row_id_X_test = row_id_X[train_index], row_id_X[test_index]
        y_col = le1.inverse_transform(sorted(set(y_train.tolist())))
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        clf.fit(X_train, y_train)
        data = pd.DataFrame(clf.predict_proba(X_test), columns=y_col)
        for k in [m for m in shop_id_list if m not in data.columns]:
            data[k] = 0
        data.index = row_id_X_test
        data['row_id'] = data.index
        data = data[a]
        train_data_order_wifi = pd.concat([train_data_order_wifi, data])
    train_data_order_wifi.to_csv(path + '\\cdata\\' + l + '\\train_order_wifi.csv',index = False)

    order_wifi_dic2 = {}
    for ind in dff2.index:
        order_wifi_dic2[dff2.loc[ind, 'row_id']] = dff2.loc[ind, 'wifi_list']

    a = pd.DataFrame(order_wifi_dic2).T
    a['row_id'] = a.index
    a = a.fillna(-999)

    pre2 = [k for k in pre if k not in a.columns and k != 'row_id']
    for k in pre2:
        a[k] = -999

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(X, Y)
    y_col = le1.inverse_transform(sorted(set(Y)))
    data = pd.DataFrame(clf.predict_proba(a[pre]), columns=y_col)
    data.index = a.index
    data['row_id'] = a['row_id']
    data.to_csv(path + '\\cdata\\' + l + '\\test_order_wifi.csv',index=False)


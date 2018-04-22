
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
path = os.getcwd()



df0 = pd.read_csv('训练数据-ccf_first_round_shop_info.csv')
df1 = pd.read_csv('训练数据-ccf_first_round_user_shop_behavior.csv')
df2 = pd.read_csv('AB榜测试集-evaluation_public.csv')

df1 = df1.merge(df0, left_on='shop_id',right_on='shop_id')
df1['row_id'] = df1.index

df1 = df1[['user_id', 'shop_id', 'time_stamp', 'mall_id', 'row_id', 'wifi_infos','longitude_x', 'latitude_x']].reset_index(drop=True)
df1.columns = ['user_id', 'shop_id', 'time_stamp', 'mall_id', 'row_id', 'wifi_infos','longitude', 'latitude']



df1['Day'] = df1['time_stamp'].apply(lambda x:int(x[8:10]))
df2['Day'] = df2['time_stamp'].apply(lambda x:int(x[8:10]))

df1['hour'] = df1['time_stamp'].apply(lambda x: int(x[11:13]))
df2['hour'] = df2['time_stamp'].apply(lambda x: int(x[11:13]))

##############星期
df1['week_day'] = df1['time_stamp'].apply(lambda x: int(datetime.strptime(x,'%Y-%m-%d %H:%M').weekday()))
df2['week_day'] = df2['time_stamp'].apply(lambda x: int(datetime.strptime(x,'%Y-%m-%d %H:%M').weekday()))

############周几与hour哈希

df1['hour_day'] = df1.apply(lambda x: '%s_%s' %(x.hour,x.week_day),axis= 1)
df2['hour_day'] = df2.apply(lambda x: '%s_%s' %(x.hour,x.week_day),axis= 1)

####小时的时间段

def split_hour(x):
    if x>=9 and x<11:
        return 0
    elif x>=11 and x<13:
        return 1
    elif x >= 11 and x <14:
        return 2
    elif x >= 14 and x <17:
        return 3
    elif x >= 17 and x <21:
        return 4
    else:
        return 5

df1['hour_cag'] = df1['hour'].apply(lambda x:split_hour(x))
df2['hour_cag'] = df1['hour'].apply(lambda x:split_hour(x))
#############


####周末与平时
df1['weekend'] = df1['week_day'] .apply(lambda x:1 if x>=5 else 0)
df2['weekend'] = df2['week_day'] .apply(lambda x:1 if x>=5 else 0)

####周末与平时与小时段的交叉

df1['week_cross_hour'] = df1.apply(lambda x: '%s_%s' %(x.hour_cag,x.weekend),axis= 1)
df2['week_cross_hour'] = df2.apply(lambda x: '%s_%s' %(x.hour_cag,x.weekend),axis= 1)



#是否有重复wifi
def if_du1(x):
    if len(x.split(';'))!= len(list(set([l.split('|')[0] for l in x.split(';')]))):
        return 1
    else:
        return 0

#分割wifilist，形成字典
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

col_df1 = ['user_id', 'shop_id', 'time_stamp', 'mall_id', 'row_id', 'wifi_infos',
       'longitude', 'latitude', 'Day', 'hour', 'week_day', 'hour_day',
       'hour_cag', 'weekend', 'week_cross_hour', 'if_du1', 'wifi_list']

#是否连接       
df1['if_con'] = df1['wifi_list'].apply(lambda x:1 if len([m for m in x if x[m][1]==1])>0 else 0)
df2['if_con'] = df2['wifi_list'].apply(lambda x:1 if len([m for m in x if x[m][1]==1])>0 else 0)

###连接的wifi的强度
df1['stre_con'] = df1.apply(lambda x:-999 if x.if_con==0 else [x.wifi_list[m][1] for m in x.wifi_list if x.wifi_list[m][1]==1][0],axis = 1)
df2['stre_con'] = df2.apply(lambda x:-999 if x.if_con==0 else [x.wifi_list[m][1] for m in x.wifi_list if x.wifi_list[m][1]==1][0],axis = 1)
###基础信息
df1['num_wifi'] = df1['wifi_list'].apply(lambda x: len([x[k][0] for k in x]))
df1['max_stgh'] = df1['wifi_list'].apply(lambda x: max([x[k][0] for k in x]))
df1['min_stgh'] = df1['wifi_list'].apply(lambda x: min([x[k][0] for k in x]))
df1['avg_stgh'] = df1['wifi_list'].apply(lambda x: np.mean([x[k][0] for k in x]))
df1['std_stgh'] = df1['wifi_list'].apply(lambda x: np.std([x[k][0] for k in x]))

df2['num_wifi'] = df2['wifi_list'].apply(lambda x: len([x[k][0] for k in x]))
df2['max_stgh'] = df2['wifi_list'].apply(lambda x: max([x[k][0] for k in x]))
df2['min_stgh'] = df2['wifi_list'].apply(lambda x: min([x[k][0] for k in x]))
df2['avg_stgh'] = df2['wifi_list'].apply(lambda x: np.mean([x[k][0] for k in x]))
df2['std_stgh'] = df2['wifi_list'].apply(lambda x: np.std([x[k][0] for k in x]))



############################计算变量的函数

#
def cal_wifi_in_shop_cnt(x):
    s = 0
    for k in x[['wifi_list']].values[0]:
        if k in shop_wifi[x['shop_id_condi']]:
            s += 1
    return s


def cal_wifi_only_in_shop_cnt(x):
    s = 0
    for k in x[['wifi_list']].values[0]:
        if k in wifi_shop:
            if len(wifi_shop[k]) == 1 and wifi_shop[k][0] == x.shop_id_condi:
                s += 1
    return s


def cal_wifi_in_shop_tfidf_max(x):
    res = []
    for k in x[['wifi_list']].values[0]:
        if k in wifi_td_idf[x.shop_id_condi]:
            res.append(wifi_td_idf[x.shop_id_condi][k])
    return res

def cal_con_wifi_shop_prob(x):
    s = 0
    con = ''
    for k in x[['wifi_list']].values[0]:
        if x[['wifi_list']].values[0][k][1]==1:
            con=k
    if con in shop_con_wifi[x.shop_id_condi]:
        s = len([k for k in shop_con_wifi[x.shop_id_condi] if k==con])/len(shop_con_wifi[x.shop_id_condi])
    return s


def if_receive_shop_con_wifi_most(x):
    s = -999
    if x.shop_id_condi in shop_con_most:
        s = 0
        for k in x[['wifi_list']].values[0]:
            if k==shop_con_most[x.shop_id_condi]:
                s = 1
    return s


def cal_if_receive_shop_con_wifi(x):
    s = -999
    if x.shop_id_condi in shop_con_most:
        s = 0
        for k in x[['wifi_list']].values[0]:
            if k in shop_con_wifi[x.shop_id_condi]:
                s = 1
    return s


def cal_dist_receive_shop_con_wifi_most(x):
    wif =  shop_con_most[x.shop_id_condi]
    stre = x[['wifi_list']].values[0][wif][0]
    return len([k for k in dist_shop_wifi_stre[x.shop_id_condi][wif] if k<=stre])/len(dist_shop_wifi_stre[x.shop_id_condi][wif])


def cal_wifi_in_shop_dist(x):
    res = []
    for k in x[['wifi_list']].values[0]:
        if k in dist_shop_wifi_stre[x.shop_id_condi]:
            stre = x[['wifi_list']].values[0][k][0]
            w_l = dist_shop_wifi_stre[x.shop_id_condi][k]
            res.append(len([m for m in w_l if m<= stre])/len(w_l))
    return res


##经纬度的中心点的计算公式
def center_geolocation(geolocations):
	x = 0
	y = 0
	z = 0
	lenth = len(geolocations)
	for lon, lat in geolocations:
		lon = radians(float(lon))
		lat = radians(float(lat))

		x += cos(lat) * cos(lon)
		y += cos(lat) * sin(lon)
		z += sin(lat)

	x = float(x / lenth)
	y = float(y / lenth)
	z = float(z / lenth)

	return (degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))))




######全部mall的表，首次要运行#####

# for d in range(18,33):
#     feature_data_all_mall = df1[(df1.Day < d)]
#
#     ###用户的记录数
#     a = pd.DataFrame(feature_data_all_mall.groupby('user_id')['shop_id'].count()).reset_index()
#     a.columns = ['user_id','user_cnt']
#     a.to_csv('feature_all_mall\\'+str(d)+'_user_cnt.csv',index=False)
#     ##用户去过最多的类型
#     a = pd.merge(feature_data_all_mall[['user_id','shop_id']],df0[['shop_id','category_id']],how='left',on = 'shop_id')
#     b = pd.DataFrame(a.groupby('user_id')['category_id'].apply(lambda x:Counter(x).most_common()[0][0])).reset_index()
#     b.columns = ['user_id','most_fre_shop_categ']
#     b.to_csv('feature_all_mall\\' + str(d) + '_most_fre_shop_categ.csv', index=False)
#
#     ###用户之前消费过的价格区间（最大，最小，平均，极差,中位数）
#
#     a = pd.merge(feature_data_all_mall[['user_id', 'shop_id']], df0[['shop_id', 'price']], how='left', on='shop_id')
#     b = pd.DataFrame(a.groupby('user_id')['price'].max()).reset_index()
#     b.columns = ['user_id', 'max_user_price']
#     b.to_csv('feature_all_mall\\' + str(d) + '_max_user_price.csv', index=False)
#
#     b = pd.DataFrame(a.groupby('user_id')['price'].min()).reset_index()
#     b.columns = ['user_id', 'min_user_price']
#     b.to_csv('feature_all_mall\\' + str(d) + '_min_user_price.csv', index=False)
#
#     b = pd.DataFrame(a.groupby('user_id')['price'].mean()).reset_index()
#     b.columns = ['user_id', 'mean_user_price']
#     b.to_csv('feature_all_mall\\' + str(d) + '_mean_user_price.csv', index=False)
#
#     b = pd.DataFrame(a.groupby('user_id')['price'].apply(lambda x:max(x)-min(x))).reset_index()
#     b.columns = ['user_id', 'exval_user_price']
#     b.to_csv('feature_all_mall\\' + str(d) + '_exval_user_price.csv', index=False)
#
#     b = pd.DataFrame(a.groupby('user_id')['price'].apply(lambda x: np.median(x))).reset_index()
#     b.columns = ['user_id', 'median_user_price']
#     b.to_csv('feature_all_mall\\' + str(d) + '_median_user_price.csv', index=False)


#
#
# df3 = df1[['Day','wifi_list']].append(df2[['Day','wifi_list']]).reset_index(drop=True)
#
# wifi_cnt_info = {}
# for ind in df3.index:
#     for k in df3.loc[ind,'wifi_list']:
#         if k not in wifi_cnt_info:
#             wifi_cnt_info[k] = {}
#             wifi_cnt_info[k]['Day'] = [df3.loc[ind,'Day']]
#             wifi_cnt_info[k]['stre'] = [df3.loc[ind,'wifi_list'][k][0]]
#         else:
#             wifi_cnt_info[k]['Day'].append(df3.loc[ind,'Day'])
#             wifi_cnt_info[k]['stre'].append(df3.loc[ind,'wifi_list'][k][0])
#
#
#
# ###wifi出现的次数
# wifi_cnt_fre = {}
# for w in wifi_cnt_info:
#     wifi_cnt_fre[w] = len(wifi_cnt_info[w]['Day'])
#
#
# a = pd.DataFrame.from_dict(wifi_cnt_fre,orient='index').reset_index()
# a.columns = ['wifi','cnt']
#
# b = pd.DataFrame(a.cnt.value_counts()).reset_index()
# output = open('wifi_cnt_fre.pkl', 'wb')
# pickle.dump(wifi_cnt_fre, output)
# output.close()
#
#
# ###wifi出现的天数
# wifi_cnt_day= {}
# for w in wifi_cnt_info:
#     wifi_cnt_day[w] = len(set(wifi_cnt_info[w]['Day']))
#
# output = open('wifi_cnt_day.pkl', 'wb')
# pickle.dump(wifi_cnt_day, output)
# output.close()
#
#
#
#
#
# ###wifi的平均强度
# wifi_cnt_mean_streng = {}
# for w in wifi_cnt_info:
#     wifi_cnt_mean_streng[w] = np.mean(wifi_cnt_info[w]['stre'])
#
# output = open('wifi_cnt_mean_streng.pkl', 'wb')
# pickle.dump(wifi_cnt_mean_streng, output)
# output.close()
#



#######################################################################################################变量计算

mall_list = list(set(df0.mall_id))
i = 0
for l in mall_list:
    print(l,i)
    i+=1

open_file = open('C:\\xww\evan_mime\\Python\\Project\\用户定位天池大赛_20171011\pkl文件\\mall_wait_shop_20171114\\pkl_'+str(l)+'.pkl', 'rb')
row_id_condi_train = pickle.load(open_file)
open_file.close()



###################加载候选集
open_file = open('houxuan\\mall_wait_shop_20171114\\pkl_'+str(l)+'.pkl', 'rb')
row_id_condi_train = pickle.load(open_file)
open_file.close()


wifiDict = {
    'row_id': [],
    'shop_id_condi': []
}

for r in row_id_condi_train:
    for s in row_id_condi_train[r]:
        wifiDict['row_id'].append(r)
        wifiDict['shop_id_condi'].append(s)

df_train = pd.DataFrame(wifiDict)



for d in range(18,32):
    print(d)
    label_data = df1[((df1.Day==d) & (df1.mall_id==l))]
    feature_data = df1[((df1.Day<d) & (df1.mall_id==l))]

##################evan################################                       
'''
计算shop有wifi连接的历史记录数，1
各记录wifi个数的平均值,1
出现次数topnwifi出现总数占比,
出现的wifi数与记录数之比,
历史记录中wifi强度的均值、最大值、最小值
连接wifi的个数与1的比值
有wifi连接的历史记录数的占比1
'''
feature_data_copy = feature_data

#计算shop有wifi连接的历史记录数,以及记录数,有wifi连接的历史记录数的占比
shop_con_records = feature_data.groupby(by='shop_id',as_index=False)['if_con'].agg({'con_record_num':np.sum,'record_num':'count'})
shop_con_records['con_record_num_ratio'] = shop_con_records['record_num']/shop_con_records['con_record_num']

#各记录wifi个数的平均值
feature_data_copy['wifi_num'] = feature_data_copy['wifi_list'].apply(lambda x:len(x.keys()))
shop_wifi_num_avg = feature_data.groupby(by='shop_id',as_index=False)['wifi_num'].agg({'wifi_num_avg':np.mean})

#shop-wifi强度的均值、最大值、最小值,连接wifi的个数,出现的wifi数
feature_data_copy_tran = feature_data_copy.drop('wifi_infos',axis=1).join(feature_data_copy.wifi_infos.str.split(';',expand=True).stack().reset_index(level=1,drop=True).rename('wifi'))
feature_data_copy_tran['bssid'] = feature_data_copy_tran['wifi'].apply(lambda x:x.split('|')[0])
feature_data_copy_tran['strength'] = feature_data_copy_tran['wifi'].apply(lambda x:int(x.split('|')[1]))
feature_data_copy_tran['connect'] = feature_data_copy_tran['wifi'].apply(lambda x:1 if x.split('|')[2]=='true' else 0)

#shop-wifi强度的均值、最大值、最小值,
shop_wifi_strinfo = feature_data_copy_tran.groupby(by='shop_id',as_index=False)['strength'].agg({'shop_wifi_str_mean':np.mean,'shop_wifi_str_max':np.max,'shop_wifi_str_min':np.min,'shop_wifi_str_std':np.std})

#shop-wifi连接wifi的个数
shop_con_wifi_num = feature_data_copy_tran.loc[feature_data_copy_tran.loc[:,'connect']==1,:].groupby(by='shop_id',as_index=False).bssid.agg({'shop_con_wifi_num':lambda x: len(x.unique())})

#shop-wifi出现wifi的个数
shop_wifi_num = feature_data_copy_tran.groupby(by='shop_id',as_index=False).bssid.agg({'shop_wifi_num':lambda x: len(x.unique())})

#合并
shop_wifi_info_group = shop_con_records.merge(shop_wifi_num_avg).merge(shop_wifi_strinfo).merge(shop_con_wifi_num).merge(shop_wifi_num)

#出现的wifi数与记录数之比
shop_wifi_info_group['shop_wifi_num_ratio'] =shop_wifi_info_group['shop_wifi_num']/shop_wifi_info_group['record_num']
shop_wifi_info_group['shop_con_wifi_num_ratio'] =shop_wifi_info_group['shop_con_wifi_num']/shop_wifi_info_group['con_record_num']

#####################################################


    ####与候选集关联

    Train = pd.merge(label_data, df_train, how = 'left', on='row_id')
    Train = Train.dropna(axis=0)
    Train['Y'] = Train.apply(lambda x: 1 if x.shop_id == x.shop_id_condi else 0, axis=1)

    #######用户之前出现的记录数(所有的mall)
    a = pd.read_csv('feature_all_mall\\'+str(d)+'_user_cnt.csv')
    Train = pd.merge(Train, a, on='user_id',how='left')

    #######用户之前的消费过最多商店类型
    b = pd.read_csv('feature_all_mall\\' + str(d) + '_most_fre_shop_categ.csv')
    Train = pd.merge(Train, b, on='user_id', how='left')


    ####用户之前消费过的价格区间（最大，最小，平均，极差）
    b = pd.read_csv('feature_all_mall\\' + str(d) + '_max_user_price.csv')
    Train = pd.merge(Train, b, on='user_id', how='left')

    b = pd.read_csv('feature_all_mall\\' + str(d) + '_min_user_price.csv')
    Train = pd.merge(Train, b, on='user_id', how='left')

    b = pd.read_csv('feature_all_mall\\' + str(d) + '_mean_user_price.csv')
    Train = pd.merge(Train, b, on='user_id', how='left')

    b = pd.read_csv('feature_all_mall\\' + str(d) + '_exval_user_price.csv')
    Train = pd.merge(Train, b, on='user_id', how='left')

    b = pd.read_csv('feature_all_mall\\' + str(d) + '_median_user_price.csv')
    Train = pd.merge(Train, b, on='user_id', how='left')

    pre = ['user_cnt','max_user_price','min_user_price','mean_user_price','exval_user_price','median_user_price']


    #####经纬度

    pre.extend(['longitude','latitude'])

    #####时间段，六个时间范围
    pre.extend(['hour', 'week_day', 'hour_day','hour_cag', 'weekend', 'week_cross_hour'])


    ####候选shop_id，出现次数与比重

    a = pd.DataFrame(feature_data.groupby('shop_id')['user_id'].count()).reset_index()
    a.columns = ['shop_id_condi','shop_cnt_fre']
    a['shop_cnt_per'] = a['shop_cnt_fre']/sum(a.shop_cnt_fre)
    Train = pd.merge(Train, a, on='shop_id_condi', how='left')
    pre.extend(['shop_cnt_per','shop_cnt_fre'])

    #####本mall与候选shop_id 类型一致shop数量
    a = dict(zip(df0.shop_id,df0.category_id))
    Train['category_id'] = Train['shop_id_condi'].apply(lambda x:a[x])


    a = df0[df0.mall_id==l][['shop_id','category_id']]
    b = pd.DataFrame(a.groupby('category_id')['shop_id'].count()).reset_index()
    b.columns = ['category_id','mall_same_category_num']

    Train = pd.merge(Train,b,how='left',on='category_id')
    pre.append('mall_same_category_num')

    ####候选shop的价格
    a = dict(zip(df0.shop_id, df0.price))
    Train['price'] = Train['shop_id_condi'].apply(lambda x: a[x])
    pre.append('price')


    ######8）是否连接wifi,连接wifi的强度,连接wifi的强度序数,wifilist的平均强度,wifilist的强度标准差,wifilist最大强度,wifilist的最小强度
    pre.extend([ 'stre_con', 'num_wifi', 'max_stgh', 'min_stgh', 'avg_stgh','std_stgh'])


##########################二级关联
    ###1）用户之前去过候选shop的次数与用户记录数占比

    a = pd.DataFrame(feature_data.groupby(['user_id','shop_id'])['Day'].count()).reset_index()
    a.columns = ['user_id','shop_id_condi','usr_cnt_shop_condi']
    Train = pd.merge(Train, a, how = 'left',on = ['user_id','shop_id_condi'])

    Train['usr_per_shop_condi'] = Train['usr_cnt_shop_condi']/Train['user_cnt']
    pre.extend(['usr_per_shop_condi','usr_cnt_shop_condi'])


    ###2)位置与候选shop_id之间的距离
    a = dict(zip(df0.shop_id,zip(df0.longitude,df0.latitude)))
    Train['loc_shop_id'] = Train['shop_id_condi'].apply(lambda x:a[x])
    Train['user_distance_shop_condi'] = Train.apply(lambda x: np.linalg.norm(np.array([x.longitude,x.latitude]) - np.array([x.loc_shop_id[0],x.loc_shop_id[1]])),axis =1)
    pre.append('user_distance_shop_condi')

    ####当前时间段（小时或者week,或者week与 早中晚午的交叉）候选shop_id记录数，比重

    ##有六种时间 1.小时，2.周几，3.小时与周几hash 4.时间段，5.周末与否，6小时段与周末与否交叉
    col = ['hour', 'week_day', 'hour_day', 'hour_cag', 'weekend', 'week_cross_hour']

    for c in col:
        a = pd.DataFrame(feature_data.groupby([c,'shop_id'])['user_id'].count()).reset_index()
        a.columns = [c,'shop_id_condi',  'shop_same_'+str(c)+'_cnt']
        Train = pd.merge(Train, a, how='left', on=[c, 'shop_id_condi'])

        a = pd.DataFrame(feature_data.groupby([c])['user_id'].count()).reset_index()
        a.columns = [c, 'cnt_'+str(c)]
        Train = pd.merge(Train, a, how='left', on=c)

        Train['shop_same_' + str(c) + '_pre'] = Train['shop_same_' + str(c) + '_cnt'] / Train['shop_cnt_fre']
        Train['shop_same_' + str(c) + '_pre_tot'] = Train['shop_same_' + str(c) + '_cnt'] / Train['cnt_' + str(c)]
        pre.extend(['shop_same_' + str(c) + '_cnt', 'shop_same_' + str(c) + '_pre', 'shop_same_' + str(c) + '_pre_tot'])


    ######当前week类，每天shop比重的  平均，最大，最小，趋势值

    col = ['weekend']

    for c in col:
        dfeature = pd.DataFrame(feature_data.groupby(['shop_id',c,'Day'])['user_id'].count()).reset_index()
        dfeature.columns = ['shop_id',str(c),'Day','cnt']

        dfeaturetot = pd.DataFrame(feature_data.groupby([c,'Day'])['shop_id'].count()).reset_index()
        dfeaturetot.columns = [str(c), 'Day', 'cnt_tot']

        datafeature = pd.merge(dfeature, dfeaturetot, how='left', on=[str(c), 'Day'])
        datafeature['per'] = datafeature['cnt']/datafeature['cnt_tot']

        b = pd.DataFrame(datafeature.groupby(['shop_id',c]).apply(lambda x:len(x))).reset_index()
        b.columns = ['shop_id',str(c),'cnt_Day_'+str(c)]
        Train = pd.merge(Train, b, left_on=[c, 'shop_id_condi'],right_on= [c, 'shop_id'])

        ###平均
        b = pd.DataFrame(datafeature.groupby(['shop_id', c])['per'].mean()).reset_index()
        b.columns = ['shop_id', str(c), 'avg_per_'+str(c)]
        Train = pd.merge(Train, b, left_on=[c, 'shop_id_condi'], right_on=[c, 'shop_id'])

        ###最大
        b = pd.DataFrame(datafeature.groupby(['shop_id', c])['per'].max()).reset_index()
        b.columns = ['shop_id', str(c), 'max_per_' + str(c)]
        Train = pd.merge(Train, b, left_on=[c, 'shop_id_condi'], right_on=[c, 'shop_id'])


        ###最小
        b = pd.DataFrame(datafeature.groupby(['shop_id', c])['per'].min()).reset_index()
        b.columns = ['shop_id', str(c), 'min_per_' + str(c)]
        Train = pd.merge(Train, b, left_on=[c, 'shop_id_condi'], right_on=[c, 'shop_id'])

        pre.extend(['cnt_Day_'+str(c),'avg_per_'+str(c),'max_per_' + str(c),'min_per_' + str(c)])


    #####每个shop的wifi集合
    shop_wifi = {}
    for ind in feature_data.index:
        if feature_data.loc[ind,'shop_id'] not in shop_wifi:
            shop_wifi[feature_data.loc[ind,'shop_id']] = {}
            for k in feature_data.loc[ind,'wifi_list']:
                shop_wifi[feature_data.loc[ind, 'shop_id']][k] = [feature_data.loc[ind, 'Day']]
        else:
            for k in feature_data.loc[ind, 'wifi_list']:
                if k not in shop_wifi[feature_data.loc[ind, 'shop_id']]:
                    shop_wifi[feature_data.loc[ind, 'shop_id']][k] = [feature_data.loc[ind, 'Day']]
                else:
                    shop_wifi[feature_data.loc[ind, 'shop_id']][k].append(feature_data.loc[ind, 'Day'])

    wifi_shop = {}
    for s in shop_wifi:
        for w in shop_wifi[s]:
            if w not in wifi_shop:
                wifi_shop[w] = [s]
            else:
                wifi_shop[w].append(s)

    ####有几个wifi出现过，

    Train['wifi_in_shop_cnt'] = Train.apply(lambda x:cal_wifi_in_shop_cnt(x),axis = 1)
    Train['wifi_only_in_shop_cnt'] = Train.apply(lambda x: cal_wifi_only_in_shop_cnt(x), axis=1)
    Train['wifi_in_shop_per'] = Train['wifi_in_shop_cnt']/Train['wifi_list'].apply(lambda x:len(x))

    pre.extend(['wifi_in_shop_cnt','wifi_only_in_shop_cnt'])


    ######wifi_list在候选shop_id的tf_idf类的变量

    wifi_td_idf = {}

    for s in shop_wifi:
        wifi_td_idf[s] = {}
        for k in shop_wifi[s]:
            wifi_td_idf[s][k] = (len(shop_wifi[s][k])/sum([len(shop_wifi[s]) for z in shop_wifi[s]]))*np.log((len(shop_wifi))/(len(wifi_shop[k])+1))


    Train['wifi_in_shop_tfidf_res'] = Train.apply(lambda x:cal_wifi_in_shop_tfidf_max(x) , axis=1)

    Train['wifi_in_shop_tfidf_max'] = Train['wifi_in_shop_tfidf_res'].apply(lambda x:0 if len(x)==0 else max(x))
    Train['wifi_in_shop_tfidf_min'] = Train['wifi_in_shop_tfidf_res'].apply(lambda x: 0 if len(x) == 0 else min(x))
    Train['wifi_in_shop_tfidf_mean'] = Train['wifi_in_shop_tfidf_res'].apply(lambda x: 0 if len(x) == 0 else np.mean(x))
    Train['wifi_in_shop_tfidf_std'] = Train['wifi_in_shop_tfidf_res'].apply(lambda x: 0 if len(x) == 0 else np.std(x))
    Train['wifi_in_shop_tfidf_med'] = Train['wifi_in_shop_tfidf_res'].apply(lambda x: 0 if len(x) == 0 else np.median(x))
    Train['wifi_in_shop_tfidf_x'] = Train['wifi_in_shop_tfidf_res'].apply(lambda x: 0 if len(x) == 0 else reduce(lambda z, zz: z * zz, x))
    pre.extend(['wifi_in_shop_tfidf_max','wifi_in_shop_tfidf_min','wifi_in_shop_tfidf_mean','wifi_in_shop_tfidf_std','wifi_in_shop_tfidf_med','wifi_in_shop_tfidf_x'])


    #######是否有连接wifi
    Train['if_con'] = Train['wifi_list'].apply(lambda x:1 if sum([x[k][1] for k in x])>0 else 0)

    shop_con_wifi = {}
    for ind in feature_data.index:
        if feature_data.loc[ind,'shop_id'] not in shop_con_wifi:
            shop_con_wifi[feature_data.loc[ind,'shop_id']] = []
            for k in feature_data.loc[ind,'wifi_list']:
                if feature_data.loc[ind,'wifi_list'][k][1]==1:
                    shop_con_wifi[feature_data.loc[ind, 'shop_id']].append(k)
        else:
            for k in feature_data.loc[ind, 'wifi_list']:
                if feature_data.loc[ind,'wifi_list'][k][1]==1:
                    shop_con_wifi[feature_data.loc[ind, 'shop_id']].append(k)
    ###连接的wifi在候选集shop_id被连接的概率（连接wifi的次数/所有的连接记录数）
    Train['con_wifi_shop_prob'] = Train.apply(lambda x: -999 if x.if_con == 0 else cal_con_wifi_shop_prob(x), axis=1)

    #####候选shop连接的记录数，与比例
    shop_con_cnt = {}
    for s in shop_con_wifi:
        shop_con_cnt[s] = len(shop_con_wifi[s])


    Train['shop_con_cnt'] = Train['shop_id_condi'].apply(lambda x: -999 if x not in shop_con_cnt else shop_con_cnt[x])
    Train['shop_con_per'] = Train['shop_con_cnt']/Train['shop_cnt_fre']




    ###shop连接次数最多的wifi的
    shop_con_most = {}
    for s in shop_con_wifi:
        if len(shop_con_wifi[s])>1:
            if Counter(shop_con_wifi[s]).most_common()[0][1]>1:
                shop_con_most[s] = Counter(shop_con_wifi[s]).most_common()[0][0]



    ###wifi中有无出现在候选集shop_id连接次数最多的top1的wifi
    Train['if_receive_shop_con_wifi_most'] = Train.apply(lambda x:if_receive_shop_con_wifi_most(x), axis=1)


    ###wifi中有无出现之前在该店铺中连接到的wifi
    Train['if_receive_shop_con_wifi'] = Train.apply(lambda x:cal_if_receive_shop_con_wifi(x), axis=1)
    pre.extend(['con_wifi_shop_prob', 'if_con','if_receive_shop_con_wifi_most','if_receive_shop_con_wifi','shop_con_cnt','shop_con_per'])


    ####shop中每个wifi的强度的分布
    dist_shop_wifi_stre = {}
    for ind in feature_data.index:
        if feature_data.loc[ind,'shop_id'] not in dist_shop_wifi_stre:
            dist_shop_wifi_stre[feature_data.loc[ind,'shop_id']] = {}
            for k in feature_data.loc[ind,'wifi_list']:
                dist_shop_wifi_stre[feature_data.loc[ind, 'shop_id']][k] = [feature_data.loc[ind,'wifi_list'][k][0]]
        else:
            for k in feature_data.loc[ind, 'wifi_list']:
                if k not in dist_shop_wifi_stre[feature_data.loc[ind, 'shop_id']]:
                    dist_shop_wifi_stre[feature_data.loc[ind, 'shop_id']][k] = [feature_data.loc[ind,'wifi_list'][k][0]]
                else:
                    dist_shop_wifi_stre[feature_data.loc[ind, 'shop_id']][k].append(feature_data.loc[ind,'wifi_list'][k][0])
    ####候选shop被最多次连接的wifi的强度分布
    Train['dist_receive_shop_con_wifi_most'] = Train.apply(lambda x:-999 if x.if_receive_shop_con_wifi_most!=1 else cal_dist_receive_shop_con_wifi_most(x), axis=1)
    pre.append('dist_receive_shop_con_wifi_most')


    #####wifi在候选shop的分布

    Train['wifi_in_shop_dist'] = Train.apply(lambda x:cal_wifi_in_shop_dist(x) , axis=1)

    Train['wifi_in_shop_dist_max'] = Train['wifi_in_shop_dist'].apply(lambda x:0 if len(x)==0 else max(x))
    Train['wifi_in_shop_dist_min'] = Train['wifi_in_shop_dist'].apply(lambda x: 0 if len(x) == 0 else min(x))
    Train['wifi_in_shop_dist_mean'] = Train['wifi_in_shop_dist'].apply(lambda x: 0 if len(x) == 0 else np.mean(x))
    Train['wifi_in_shop_dist_std'] = Train['wifi_in_shop_dist'].apply(lambda x: 0 if len(x) == 0 else np.std(x))
    Train['wifi_in_shop_dist_med'] = Train['wifi_in_shop_dist'].apply(lambda x: 0 if len(x) == 0 else np.median(x))
    Train['wifi_in_shop_dist_x'] = Train['wifi_in_shop_dist'].apply(lambda x: 0 if len(x) == 0 else reduce(lambda z, zz: z * zz, x))
    pre.extend(['wifi_in_shop_dist_max','wifi_in_shop_dist_min','wifi_in_shop_dist_mean','wifi_in_shop_dist_std','wifi_in_shop_dist_med','wifi_in_shop_dist_x'])



    ########候选shop记录经纬度的平均值

    ##每个shop出现的经纬度(只计算正常经纬度的值)
    rang_la_lo = [df0.longitude[df0.mall_id == l].max()+0.00001,df0.longitude[df0.mall_id == l].min()-0.00001,df0.latitude[df0.mall_id == l].max()+0.00001,df0.latitude[df0.mall_id == l].min()-0.00001]

    ####shop_id历史出现的经纬度
    shop_la_lo_real = {}
    for ind in feature_data.index:
        if feature_data.loc[ind,'shop_id'] not in shop_la_lo_real:
            if feature_data.loc[ind,'longitude']>= rang_la_lo[1] and feature_data.loc[ind,'longitude']<rang_la_lo[0] and feature_data.loc[ind,'latitude']>= rang_la_lo[3] and feature_data.loc[ind,'latitude']<rang_la_lo[2]:
                shop_la_lo_real[feature_data.loc[ind,'shop_id']] = [[feature_data.loc[ind,'longitude'],feature_data.loc[ind,'latitude']]]
        else:
            if feature_data.loc[ind,'longitude']>= rang_la_lo[1] and feature_data.loc[ind,'longitude']<rang_la_lo[0] and feature_data.loc[ind,'latitude']>= rang_la_lo[3] and feature_data.loc[ind,'latitude']<rang_la_lo[2]:
                shop_la_lo_real[feature_data.loc[ind,'shop_id']].append([feature_data.loc[ind,'longitude'],feature_data.loc[ind,'latitude']])

    ####shop_id的历史的中心点

    shop_la_lo_real_zhongxin = {}
    for s in shop_la_lo_real:
        shop_la_lo_real_zhongxin[s] = center_geolocation(shop_la_lo_real[s])

    Train['shop_id_zhongxin_history_long'] = Train['shop_id_condi'].apply(lambda x: -999 if x not in shop_la_lo_real_zhongxin else shop_la_lo_real_zhongxin[x][0])
    Train['shop_id_zhongxin_history_lat'] = Train['shop_id_condi'].apply(lambda x: -999 if x not in shop_la_lo_real_zhongxin else shop_la_lo_real_zhongxin[x][1])

    Train['shop_id_max_history_long'] = Train['shop_id_condi'].apply(lambda x: -999 if x not in shop_la_lo_real else max([k[0] for k in shop_la_lo_real[x]]))
    Train['shop_id_min_history_long'] = Train['shop_id_condi'].apply(lambda x: -999 if x not in shop_la_lo_real else min([k[0] for k in shop_la_lo_real[x]]))

    Train['shop_id_max_history_lat'] = Train['shop_id_condi'].apply(lambda x: -999 if x not in shop_la_lo_real else max([k[1] for k in shop_la_lo_real[x]]))
    Train['shop_id_min_history_lat'] = Train['shop_id_condi'].apply(lambda x: -999 if x not in shop_la_lo_real else min([k[1] for k in shop_la_lo_real[x]]))

####当前当前距离候选shop中心点的欧氏距离
    Train['far_to_zhongxin'] = Train.apply(lambda x: np.linalg.norm(np.array([x.longitude,x.latitude]) - np.array([x.shop_id_zhongxin_history_long,x.shop_id_zhongxin_history_lat])),axis =1)

    ###

    pre.extend(['shop_id_zhongxin_history_long','shop_id_zhongxin_history_lat','shop_id_max_history_long','shop_id_min_history_long','shop_id_max_history_lat','shop_id_min_history_lat','far_to_zhongxin'])


    Train.to_csv(path + '\\data\\' + l + '\\'+str(d)+'dtrain.csv', index=False)




#训练集



Dtrain = pd.DataFrame()
for d in range(18,28):
    data = pd.read_csv(path + '\\data\\' + l + '\\'+str(d)+'dtrain.csv')
    Dtrain = Dtrain.append(data)




Dtest = pd.DataFrame()

for d in range(28,32):
    data = pd.read_csv(path + '\\data\\' + l + '\\'+str(d)+'dtrain.csv')
    Dtest = Dtest.append(data)


####类别变量的label_hot_编码
le1 = preprocessing.LabelEncoder()
le1.fit(list(set(df1.hour_day)))
Dtrain['hour_day'] = le1.transform(Dtrain['hour_day'] )
Dtest['hour_day'] = le1.transform(Dtest['hour_day'] )

le1 = preprocessing.LabelEncoder()
le1.fit(list(set(df1.week_cross_hour)))
Dtrain['week_cross_hour'] = le1.transform(Dtrain['week_cross_hour'] )
Dtest['week_cross_hour'] = le1.transform(Dtest['week_cross_hour'] )




X = Dtrain[pre]
Y = Dtrain['Y']
dtrain = xgb.DMatrix(X, Y)


X_ = Dtest[pre]
Y_ = Dtest['Y']

dtest = xgb.DMatrix(X_, Y_)


params = {
    'objective': 'binary:logistic',
    'eta': 0.08,
    'max_depth': 9,
    'eval_metric': 'error',
    'seed': 0,
    'missing': -999,
    'silent': 1,
    'lambda':3
}



watchlist = [(dtrain, 'train'), (dtest, 'val')]

num_rounds=1000
model = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=30)

##, early_stopping_rounds=20

####测试集



Dtest['res']=model.predict(dtest)


a = Dtest.groupby(['row_id']).apply(lambda x: dict(zip(x.shop_id_condi, x.res)))

b = pd.DataFrame(a).reset_index()
b.columns = [['row_id', 'pre']]
b['pre'] = b['pre'].apply(lambda x: max(x, key=lambda k: x[k]))


c = Dtest[['row_id','shop_id_x']].reset_index(drop=True)
c = c.drop_duplicates()

z = pd.merge(c,b,on='row_id')


sum(z.shop_id_x==z.pre)/len(z)

#### 0.73094170403587444
####0.75254790052996334

#####0.76355483081940478

###有点蒙圈 看下有没有加错误的变量

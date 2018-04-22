#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:10:26 2017
我也不知道这份多少分 这份是线下的，0.9153  线上大概0.928到0.930的二分类单模型？ 我忘了  把文件放进来 直接能跑 直接出分 我没有github 
语文不是很好 看不懂的请别骂我  那种取名rua的特征 你们看看就行 架子直接拿赤子膜拜的架子改的 我自己的我自己都看不懂 感谢他 感谢煎饼和data蝙蝠侠 （：
@author: gyp
"""

# -*- coding: utf-8 -*
import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from collections import Counter

cache_path = 'dingwei_cache1/'
train_path = 'train.csv'
shop_path='shop.csv'
wifi_shop_info=pd.DataFrame()
wifi_shop_max=pd.DataFrame()
wifi_shop_min=pd.DataFrame()
wifi_shop_connect_only=pd.DataFrame()
shop_loc=pd.DataFrame()
wifi_shop_count=pd.DataFrame()
shop_wifi={}
tx={}
tx_min={}
flag = False
def get_time_hour(data):
    data['hourofday']=pd.DatetimeIndex(data.time_stamp).hour
    data['dayofweek']=pd.DatetimeIndex(data.time_stamp).day
    return data
# 相差的分钟数
def diff_of_minutes(time1, time2):
    d = {'5': 0, '6': 31, } #????
    try:
        days = (d[time1[6]] + int(time1[8:10])) - (d[time2[6]] + int(time2[8:10]))
        try:
            minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
        except:
            minutes1 = 0
        try:
            minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
        except:
            minutes2 = 0
        return (days * 1440 - minutes2 + minutes1)
    except:
        return np.nan

# 计算两点之间距离
def call_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L


def call_orientation(lat1,lon1,lat2,lon2): 
    radLat1 = math.radians(lat1)  
    radLon1 = math.radians(lon1)  
    radLat2 = math.radians(lat2)  
    radLon2 = math.radians(lon2)  
    dLon = radLon2 - radLon1  
    y = math.sin(dLon) * math.cos(radLat2)  
    x = math.cos(radLat1) * math.sin(radLat2) - math.sin(radLat1) * math.cos(radLat2) * math.cos(dLon)  
    brng = math.degrees(math.atan2(y, x))  
    brng = (brng + 360) % 360  
    return brng


#生成binary-label
def get_label(data): #data:DataFrame
    true = dict(zip(train['row_id'].values, train['shop_id'])) #row_id对应shop_id
    print ('label map finish')
    data['label'] = data['row_id'].map(true) #map到每个shop_id
    data['label'] = (data['label'] == data['shop_id']).astype('int') #生成binary-y
    return data
#生成shop_id对应的wifi强度	
def make_wifi_shop_relation(train):
    fuck=train.values #shop_id-wifi_infos
    res1=[]
    res2=[]
    res3=[]
    for i in fuck:
        wifis=i[1].split(';') #拆分wifi_infos
        for j in wifis:
            wifi=j.split('|')
            res1.append(i[0]) #shop_id
            res2.append(wifi[0]) #bssid
            res3.append(int(wifi[1])) #power
    t_data=pd.DataFrame()
    t_data['shop_id']=res1
    t_data['wifi_id']=res2
    t_data['power']=res3
    return t_data
#生成shop_id对应row_id对应wifi强度
def f_owen(train):
    fuck=train.values
    res1=[]
    res2=[]
    res3=[]
    res4=[]
    for i in fuck:
        wifis=i[1].split(';')
        for j in wifis:
            wifi=j.split('|')
            res4.append(i[2])
            res1.append(i[0])
            res2.append(wifi[0])
            res3.append(int(wifi[1]))
    t_data=pd.DataFrame()
    t_data['shop_id']=res1
    t_data['wifi_id']=res2
    t_data['power']=res3
    t_data['row_id']=res4
    return t_data

#生成row_id对应wifi强度     
def make_wifi_rowid_relation(test):
    fuck=test.values
    res1=[]
    res2=[]
    res3=[]
    for i in fuck:
        wifis=i[1].split(';')
        for j in wifis:
            wifi=j.split('|')
            res1.append(i[0])
            res2.append(wifi[0])
            res3.append(int(wifi[1]))
    t_data=pd.DataFrame()
    t_data['row_id']=res1
    t_data['wifi_id']=res2
    t_data['power']=res3
    return t_data  

#生成shop_id对应的wifi连接	
def wifi_shop_connect(train):
    fuck=train[['shop_id','wifi_infos']].values
    res1=[]
    res2=[]
    res3=[]
    for i in fuck:
        wifis=i[1].split(';')
        for j in wifis:
            wifi=j.split('|')
            res1.append(i[0])
            res2.append(wifi[0])
            if wifi[2]=='true':
               res3.append(1)
            else:
               res3.append(0)
    t_data=pd.DataFrame()
    t_data['shop_id']=res1
    t_data['wifi_id']=res2
    t_data['connect']=res3
    t_data=t_data[t_data.connect==1]
    return t_data  
#生成row-id对应的wifi连接	
def wifi_rowid_conenct(test):
    fuck=test[['row_id','wifi_infos']].values
    res1=[]
    res2=[]
    res3=[]
    for i in fuck:
        wifis=i[1].split(';')
        for j in wifis:
            wifi=j.split('|')
            res1.append(i[0])
            res2.append(wifi[0])
            if wifi[2]=='true':
               res3.append(1)
            else:
               res3.append(0)
    t_data=pd.DataFrame()
    t_data['row_id']=res1
    t_data['wifi_id']=res2
    t_data['connect']=res3
    t_data=t_data[t_data.connect==1]
    return t_data        
####################构造负样本##################
#构造wifishop对应信息
def get_wifi_shop_info(train):
    wifi_train = train[['shop_id','wifi_infos']].drop_duplicates()#针对这两列去重
    wifi_shop_info=make_wifi_shop_relation(wifi_train) #生成shop_id-bssid-power的df
    wifi_shop_info=wifi_shop_info.groupby(['shop_id','wifi_id'],as_index=False)['power'].agg({'ave_power':'mean'}) #计算某bssid在这个shop的平均强度
    return wifi_shop_info
    
def get_wifi_shop_max(train):
    wifi_train = train[['shop_id','wifi_infos']].drop_duplicates()
    wifi_shop_max=make_wifi_shop_relation(wifi_train)
    wifi_shop_max=wifi_shop_max.groupby(['shop_id','wifi_id'],as_index=False)['power'].agg({'max_power':'max'}) #最大强度
    return wifi_shop_max

def get_wifi_shop_min(train):
    wifi_train = train[['shop_id','wifi_infos']].drop_duplicates()
    wifi_shop_min=make_wifi_shop_relation(wifi_train)
    wifi_shop_min=wifi_shop_min.groupby(['shop_id','wifi_id'],as_index=False)['power'].agg({'min_power':'min'}) #最小强度
    return wifi_shop_min


def get_wifi_shop_count(train):
    wifi_train = train[['shop_id','wifi_infos']].drop_duplicates()
    wifi_shop_count=make_wifi_shop_relation(wifi_train)
    wifi_shop_count=wifi_shop_count.groupby(['shop_id','wifi_id'],as_index=False)['power'].agg({'wifi inshopcount':'count'}) #出现记录数，但是去重了(影响有限)？
    wifi_shop_count.sort_values(['shop_id','wifi inshopcount'],inplace=True)
    wifi_shop_count= wifi_shop_count.groupby('shop_id').tail(50) #这一步在干啥？,取出现次数排名靠前的50个bssid
    return wifi_shop_count
    

# 通过wifi来构建负样本
#先取train中单条记录中前度排名靠前的wifi，然后groupby拿到各wifi下shop出现的数量；
#同时test中取强度排名靠前的几个wifi，并与上一步的数据merge，则取到了这些wifi下对应的shop_id和出现的数量；
#对上一步数据去重，取出现数量最多的shop_id作为候选集
def get_wifi_shop(train,test):
    result_path = cache_path + 'wifi_shop_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        t1= train[['shop_id','wifi_infos','row_id']].drop_duplicates()
        t1=f_owen(t1) #生成shop_id对应row_id对应wifi强度
        t2=test[['row_id','wifi_infos']].drop_duplicates()
        t2=make_wifi_rowid_relation(t2)  #生成row-id对应的wifi强度
        dfx=t1.sort_values(['row_id','power'],ascending=False).groupby(['row_id'],as_index=False).head(2) #在row-id中取强度靠前的前两个wifi，即单条记录靠前的前两个wifi
    #可信wifi中bssid-shop适配
        dfx=dfx.groupby(['wifi_id','shop_id'],as_index=False).power.agg({'bsCount':'count'}).sort_values(['wifi_id','bsCount'],ascending=False)#即在前一步筛选后，计算每个shop_id下，bssid出现的数量
        dfx=dfx.groupby(['wifi_id'],as_index=False).head(5)#取每个bssid下出现前5的shop
    #测试集中前几wifi可用
        dfy=t2.sort_values(['row_id','power'],ascending=False).groupby(['row_id'],as_index=False).head(3)#取每个row_id下强度前3的wifi
    #所有可能解，即test中bssid对应到train中的shop_id
        dfz=pd.merge(dfx,dfy,how='right',on='wifi_id')
    #去重效果
        dfz=dfz.sort_values(['row_id','shop_id','bsCount'],ascending=False).groupby(['row_id','shop_id'],as_index=False).head(1) #即取出现次数较多的shop_id
        dfz=dfz[['row_id','shop_id']]
        result=dfz.sort_values('row_id').reset_index(drop=True)
        result = result[(~result['shop_id'].isnull())]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


#即利用lbs信息建立knn，将前6的记录作为候选集
def GetGeoCandidate(dfTrain,dfTest):
    result_path = cache_path + 'user_end_shop_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        from sklearn import neighbors
        ggDict={'row_id':[],'shop_id':[],'geoR':[]}
        dfg=pd.DataFrame(ggDict)
        dfTrain=pd.merge(dfTrain,shop[['shop_id','mall_id']],on=['shop_id'],how='left') #将train中的记录对应到mall-id
        mallList=dfTrain.mall_id.unique() #取mall_set
        for mall in mallList:
            tempTrain=dfTrain[dfTrain.mall_id==mall]
            tempTest=dfTest[dfTest.mall_id==mall]
         #经纬度候选
            n=6
            knn=neighbors.KNeighborsClassifier(n_neighbors=n,weights='uniform',metric='manhattan')
            kc=knn.fit(tempTrain[['longitude','latitude']].values,tempTrain.shop_id.values)
            neiborList=kc.kneighbors(tempTest[['longitude','latitude']].values,return_distance=False)
        #转为df
            resultDic={'row_id':[],'shop_id':[],'geoR':[]}
            shopIndex=list(tempTrain.columns).index('shop_id')#即shop_id在columns中的索引值
            for i in range(n):
               resultDic['row_id']+=list(tempTest.row_id.values)
               resultDic['shop_id']+=[tempTrain.iloc[x[i],shopIndex] for x in neiborList]  #取前x的knn返回记录
               resultDic['geoR']+=[n-i]*len(tempTest) #即上面记录的排名
            geoResult=pd.DataFrame(resultDic)
            dfg=pd.concat([dfg,geoResult],axis=0)
        result=dfg.sort_values(['row_id','geoR'],ascending=False).groupby(['row_id','shop_id'],as_index=False).head(1) #去重
        result = result[(~result['shop_id'].isnull())]
        result=result[['row_id','shop_id']]
    #df=MergeCandidate(df,dfg)
        return result
       



#这里其实是一个组合候选集，包括出现数量，数量排序，强度，及强度排序
#这边都做了归一化
def GetLCSCandidate(train,test):
    #LCS
    #参考序列
    #bssid-shop的热度
    t1= train[['shop_id','wifi_infos']].drop_duplicates()
    t1=make_wifi_shop_relation(t1)
    t2=test[['row_id','wifi_infos']].drop_duplicates()
    t2=make_wifi_rowid_relation(t2)  
    dfx=t1.groupby(['shop_id','wifi_id'],as_index=False).power.agg({'bsCount':'count','bsMedian':'median'})#生成出现次数和强度中位数的groupby数据
    dfx['srx']=dfx.groupby('shop_id').bsCount.rank(ascending=False,method='min')#生成rank字段，倒序
    dfx.srx=-dfx.srx 
    #归一化，平衡shop的交易量差异问题
    dft=dfx.groupby(['shop_id'],as_index=False).bsCount.agg({'bsMax':'max','bsMin':'min'})
    dfx=pd.merge(dfx,dft,how='left',on='shop_id')
    dft=dfx.groupby(['shop_id'],as_index=False).srx.agg({'srxMax':'max','srxMin':'min'})
    dfx=pd.merge(dfx,dft,how='left',on='shop_id')
    dfx['bsRate']=(dfx.bsCount-dfx.bsMin)/(dfx.bsMax-dfx.bsMin)
    dfx['srxRate']=(dfx.srx-dfx.srxMin)/(dfx.srxMax-dfx.srxMin)
    #删除无用行
    dfx=dfx.drop(['srx','bsMax','bsMin','srxMax','srxMin','bsCount'],axis=1)
    #记录序列————wifi强度和强度的Rank
    #去重
    dfy=t2.groupby(['row_id','wifi_id'],as_index=False).head(1)
    dfy['sry']=dfy[['row_id','wifi_id','power']].groupby('row_id').power.rank(ascending=False,method='min') #排序
    dfy.sry=-dfy.sry
    #归一化，平衡记录内部
    dft=dfy.groupby(['row_id'],as_index=False).sry.agg({'sryMax':'max','sryMin':'min'})
    dfy=pd.merge(dfy,dft,how='left',on='row_id')
    dfy['sryRate']=(dfy.sry-dfy.sryMin)/(dfy.sryMax-dfy.sryMin)
    #删除无用行
    dfy=dfy.drop(['sryMax','sryMin','sry'],axis=1)
    
    #合并
    dfz=pd.merge(dfy,dfx,how='left',on='wifi_id')
    del dfx,dfy;gc.collect()
    #如果数据量太大，在此处筛选一波dfz
    #强度差值及归一化，平衡不同bssid-shopid组合强度差的差异
    dfz['ds']=-abs(dfz.power-dfz.bsMedian)
    dft=dfz.groupby(['wifi_id','shop_id'],as_index=False).ds.agg({'dsMax':'max','dsMin':'min'})
    dfz=pd.merge(dfz,dft,how='left',on=['wifi_id','shop_id'])
    dfz['dsRate']=(dfz.ds-dfz.dsMin)/(dfz.dsMax-dfz.dsMin)
    dfz=dfz.drop(['dsMax','dsMin'],axis=1)
    i,j,m,n=(2,2,0,4)
    dfz['sr']=i*dfz.srxRate+j*dfz.bsRate+m*dfz.sryRate+n*dfz.dsRate
    dfz=dfz.groupby(['row_id','shop_id'],as_index=False).sr.agg({'lcs':'sum'}).sort_values(['row_id','lcs'],ascending=False)
    result=dfz.sort_values(['row_id','lcs'],ascending=False).groupby(['row_id'],as_index=False).head(5)
    result = result[(~result['shop_id'].isnull())]
    result=result[['row_id','shop_id']]
    return result

#用户曾经去过哪些shop
def get_user_end_shop(train,test):
    result_path = cache_path + 'user_end_shop_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        train_new=pd.merge(train,shop1,on='shop_id',how='left')
        user_shop=train_new[['shop_id','user_id','mall_id']].drop_duplicates()
        result=pd.merge(test[['user_id','mall_id','row_id']],user_shop,on=['user_id','mall_id'])
        result=result[['row_id','shop_id']].drop_duplicates()
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



























        




#用户特征
# 获取用户历史行为次数
#用户出现的记录数
def get_user_count(train,result):
    user_count = train.groupby('user_id',as_index=False)['shop_id'].agg({'user_count':'count'})
    result = pd.merge(result,user_count,on=['user_id'],how='left')
    return result

#hot of shop
#候选shop的记录数
def get_shop_hot(train,result):
    shop_hot=train.groupby('shop_id',as_index=False)['row_id'].agg({'shop_hot':'count'})
    result = pd.merge(result,shop_hot,on=['shop_id'],how='left')
    return result

# 获取用户到某商店的次数 
def get_user_shop_count(train, result):
    user_shop_count = train.groupby(['user_id','shop_id'],as_index=False)['user_id'].agg({'user_shop_count':'count'})
    result = pd.merge(result,user_shop_count,on=['user_id','shop_id'],how='left')
    return result



#商店特征
#商店属性 商店所属类别，人均消费指数 所属商场号
def get_shop_info(result):
    result=pd.merge(result,shop,on=['shop_id'],how='left')
    result['category_id']=result['category_id'].fillna(0)
    result['category']=result['category_id'].map(lambda x:'0' if x==0 else str(x)[2:]).astype('int')
    return result

   
#用户到某个类型的商店去过几次 是否去过这个类型的商店
def get_uesr_kind(train,result):
    user_kind_count=train.groupby(['user_id','category_id'],as_index=False)['category_id'].agg({'user_kind_count':'count'})
    result=pd.merge(result,user_kind_count,on=['user_id','category_id'],how='left')
    return result

#wifi特征
#有没有连接这个wifi
#这个wifi的
#推荐的店铺和最强 第二强 第三强 第四强wifi的值 现实值 和差值
#强度排名
#取记录中的第一强
def f1(x):
    wifipower={}
    wifi=[]
    power=[]
    wifis=x.split(';')
    for i in wifis:
        s=i.split('|')
        wifi.append(s[0])
        power.append(int(s[1]))
    wifipower = dict(zip(wifi, power))
    wifipower=sorted(wifipower.items(),key=lambda x:x[1],reverse=True)
    return wifipower[0]

#取记录中的第二强
def f2(x):
    wifipower={}
    wifi=[]
    power=[]
    wifis=x.split(';')
    for i in wifis:
        s=i.split('|')
        wifi.append(s[0])
        power.append(int(s[1]))
    wifipower = dict(zip(wifi, power))
    wifipower=sorted(wifipower.items(),key=lambda x:x[1],reverse=True)
    if(len(wifipower)>1): 
       return wifipower[1]
    else: 
        return 0

#即test中最强wifi(为例)，统计这种情况下，在train中，最强wifi对应的候选shop的数量 
def get_wifi(result,train):
    result['wifi1']=result['wifi_infos'].map(f1)
    result['wifi2']=result['wifi_infos'].map(f2)
    result['wifi1_id']=result['wifi1'].map(lambda x:x[0]) #取第一强的bssid
    t1= train[['shop_id','wifi_infos','row_id']].drop_duplicates()
    t1=f_owen(t1) ##生成shop_id对应row_id对应wifi强度
    '''
    t6=t1.copy()
    t6['power']=t6['power'].map(lambda x:1.0/abs(x))
    t6=t6.groupby('wifi_id','shop_id'],as_index=False)['power'].agg({'pw':'sum'})
    
    '''
    t2=t1[['shop_id','row_id','power']].groupby(['row_id','shop_id'],as_index=False)['power'].agg({'power':max})#取记录中的最强的强度
    t3=pd.merge(t2,t1,on=['shop_id','row_id','power'],how='left') #保留最强的那条记录
    t4=t3[['shop_id','row_id','wifi_id']].groupby(['wifi_id','shop_id'],as_index=False)['row_id'].agg({'topcount':'count'}) #该情况下，wifi与shop共同出现的次数
    t4.rename(columns={'wifi_id':'wifi1_id'},inplace=True)
    result=pd.merge(result,t4,on=['shop_id','wifi1_id'],how='left') #特征即共同出现次数
    result['wifi1_power']=result['wifi1'].map(lambda x:x[1])
    result['wifi2_id']=result['wifi2'].map(lambda x:0 if x==0 else x[0])
    result['wifi2_power']=result['wifi2'].map(lambda x:0 if x==0 else x[1])
    wifi_shop_info.rename(columns={'wifi_id':'wifi1_id'},inplace=True)
    result=pd.merge(result,wifi_shop_info,on=['shop_id','wifi1_id'],how='left')
    result.rename(columns={'ave_power':'ave_power1'},inplace=True)
    result['ave_power1']=result['ave_power1'].fillna(result['ave_power1'].min())
    result['power_dif1']=abs((result['ave_power1']-result['wifi1_power']).astype(np.float))
    wifi_shop_info.rename(columns={'wifi1_id':'wifi2_id'},inplace=True)
    result=pd.merge(result,wifi_shop_info,on=['shop_id','wifi2_id'],how='left')
    wifi_shop_info.rename(columns={'wifi2_id':'wifi_id'},inplace=True)
    result.rename(columns={'ave_power':'ave_power2'},inplace=True)
    result['ave_power2']=result['ave_power2'].fillna(result['ave_power2'].min())
    result['power_dif2']=abs((result['ave_power2']-result['wifi2_power']).astype(np.float))
    return result

#生成wifi_list
def wifi_name(x):
    wifi=[]
    wifis=x.split(';')
    for i in wifis:
        s=i.split('|')
        wifi.append(s[0])
    return wifi
 
def fuck(x):
    x=list(x.split(':'))
    return x

def get_same_relation(result):
    result['w1']=result['wifi_infos'].map(wifi_name)
    t2 = wifi_shop_count.groupby('shop_id')['wifi_id'].agg(lambda x:':'.join(x)).reset_index()
    t2['w2']=t2['wifi_id'].map(fuck)
    del t2['wifi_id']
    result=pd.merge(result,t2,on=['shop_id'],how='left')
    a=list(result['w1'])
    b=list(result['w2'])
    c=[]
    for i in range(len(a)):
        try:
            d=len(set(a[i])&set(b[i]))
            c.append(d)
        except:
            c.append(0)
    result['w3']=c
    #del result['w1']
    #del result['w2']
    return result

def get_wifinoshow(x,y):
    count=0
    try:      
       for i in x:
           if i in y:
              count=count+1
           else:
               count=count+0
       if len(y)>10:
           return 10-count
       else:
           return len(y)-count 
    except:
       return 10
    
def get_wifiinrat(x,y):
    count=0
    try:      
       for i in x:
           if i in y:
              count=count+1
           else:
               count=count+0
       if len(y)>10:
           return float(count)/10
       else:
           return float(count)/len(y) 
    except:
       return 0        
def get_same(x,y):
    count=0
    try:      
       for i in x:
           if i in y:
              count=count+1
           else:
               count=count+0
    except:
       return 0            
    
def get_top10_sameresult(train,result):
    t1=train.groupby('shop_id',as_index=False)['row_id'].agg({'shop_hot':'count'})
    t2=pd.merge(wifi_shop_count,t1,on='shop_id',how='left')
    t2['wifiratio']=t2['wifi inshopcount']/(t2['shop_hot'].astype(float))
    t3= train[['shop_id','wifi_infos']].drop_duplicates()
    t3=make_wifi_shop_relation(t3)
    t3=t3.groupby('wifi_id',as_index=False)['shop_id'].agg({'wifi_hot':'count'})
    t2=pd.merge(t2,t3,on='wifi_id',how='left')
    t2['wifi2ratio']=t2['wifi inshopcount']/(t2['wifi_hot'].astype(float))
    
    t4=t2.copy()
    t4=t4[t4['wifiratio']>=0.5]
    t4['p']=t4['wifi_id']+':'+t4['wifi2ratio'].astype('str')    
    t4=t4[['shop_id','p']]
    t4 = t4.groupby('shop_id')['p'].agg(lambda x:'|'.join(x)).reset_index()
    t4['p']=t4['p'].map(f_s)
    result = pd.merge(result,t4,on = ['shop_id'],how = 'left')   
    result['wifinoshow'] = map(lambda x, y: get_wifinoshow(x, y) , result['w1'], result['p'])   
    result['wifiinrat'] = map(lambda x, y: get_wifiinrat(x, y) , result['w1'], result['p']) 
    result['wifitopcount'] = map(lambda x, y: get_same(x, y) , result['w1'], result['p'])    
    t2['turerat']=t2['wifi2ratio']*t2['wifiratio']
    t2['v']=t2['wifi_id']+':'+t2['turerat'].astype('str')
    t2=t2[['shop_id','v']]
    t2 = t2.groupby('shop_id')['v'].agg(lambda x:'|'.join(x)).reset_index()
    t2['v']=t2['v'].map(f_s)
    result = pd.merge(result,t2,on = ['shop_id'],how = 'left')
    result['wifiturerat'] = map(lambda x, y: get_wifipro(x, y) , result['w1'], result['v'])
    del result['v']
    #del result['p']
    return result   

   
    

def f_s(x):
    wificount={}
    wifi=[]
    count=[]
    wifis=x.split('|')
    for i in wifis:
        s=i.split(':')
        wifi.append(s[0])
        count.append(float(s[1]))
    wificount = dict(zip(wifi, count))
    return wificount
def get_wifipro(x,y):
    count=0
    try:      
       for i in x:
           if i in y:
              count=count+float(y[i])
           else:
               count=count+0
       return count
    except:
       return 0
def get_wifimulti(x,y):
    count=1
    try:
        for i in x:
            if i in y:                
               count=count*(1-y[i])
            else:
                count=count
        return count
    except:
        return 1
def shop_pro(train,result):
   t1= train[['shop_id','wifi_infos']].drop_duplicates()
   t1=make_wifi_shop_relation(t1)
   t1=t1.groupby('wifi_id',as_index=False)['shop_id'].agg({'wifi_hot':'count'})
   t2=pd.merge(wifi_shop_count,t1,on=['wifi_id'],how='left')
   t2['pro']=t2['wifi inshopcount'].astype('float')/t2['wifi_hot']   
   t2['s']=t2['wifi_id']+':'+t2['pro'].astype('str')
   t2=t2[['shop_id','s']]
   t2 = t2.groupby('shop_id')['s'].agg(lambda x:'|'.join(x)).reset_index()
   t2['s']=t2['s'].map(f_s)
   result = pd.merge(result,t2,on = ['shop_id'],how = 'left')
   result['wifiprosum'] = map(lambda x, y: get_wifipro(x, y) , result['w1'], result['s'])
   result['wifipromulti']=map(lambda x, y: get_wifimulti(x, y) , result['w1'], result['s'])  
   #del result['s']
   return result
def fuck3(x):
    d=[]
    for i in x:
        if i in shop_wifi:
           for j in shop_wifi[i]:
               d.append(j)
    return d
    
def fuck4(x,y):
    i=Counter(y).most_common(20)
    try:
        return i[x]
    except:
        return 0
    
    
    
    
    
def get_shop_score(train,result):
    global shop_wifi
    t1= train[['shop_id','wifi_infos']].drop_duplicates()
    t1=make_wifi_shop_relation(t1)
    t1=t1.groupby('wifi_id',as_index=False)['shop_id'].agg({'wifi_hot':'count'})
    t2=pd.merge(wifi_shop_count,t1,on=['wifi_id'],how='left')
    t2['pro']=t2['wifi inshopcount'].astype('float')/t2['wifi_hot']  
    t2.sort_values(['wifi_id','pro'],inplace=True)
    t2=t2.groupby('wifi_id').tail(10)
    t2[['wifi_id','shop_id']]
    m=t2[['shop_id','wifi_id']].values
    shop_wifi=dict()
    for i in m:
       if i[1] not in shop_wifi:
          shop_wifi[i[1]]=set()
       shop_wifi[i[1]].add(i[0])  
    result['shop_score'] =result['w1'].map(fuck3)
    result['scorewifi']=map(lambda x, y: fuck4(x, y) , result['shop_id'], result['shop_score'])
    del result['shop_score']
    return result 

#组内比例
def _featureInGroupSingle(df,col):
    nameSum='_sum'
    nameRate='_groupRate'
    dfx=df.groupby('row_id',as_index=False)[col].agg({col+nameSum:'sum'})
    dfx=pd.merge(df,dfx,how='left',on='row_id')
    dfx[col+nameRate]=dfx[col]/dfx[col+nameSum]
    return dfx[['row_id',col+nameRate]]
#注：组内Rate可以，组内Rank无效
def featureInGroup(df,cols):
    afterCol=[]
    nameRate='_groupRate'
    for col in cols:
        #print('>>>>>',col)
        dfy=_featureInGroupSingle(df[['row_id',col]],col)
        df[col+nameRate]=dfy[col+nameRate]
        afterCol.append(col+nameRate)
    return df,afterCol


def featureLCS(train,test,result):
    t1= train[['shop_id','wifi_infos']].drop_duplicates()
    t1=make_wifi_shop_relation(t1)
    t2=test[['row_id','wifi_infos']].drop_duplicates()
    t2=make_wifi_rowid_relation(t2)  
    dfx=t1.groupby(['shop_id','wifi_id'],as_index=False).power.agg({'bsCount':'count','bsMedian':'median'})
    dfx['srx']=dfx.groupby('shop_id').bsCount.rank(ascending=False,method='min')
    dfx.srx=-dfx.srx
    #归一化，平衡shop的交易量差异问题
    dft=dfx.groupby(['shop_id'],as_index=False).bsCount.agg({'bsMax':'max','bsMin':'min'})
    dfx=pd.merge(dfx,dft,how='left',on='shop_id')
    dft=dfx.groupby(['shop_id'],as_index=False).srx.agg({'srxMax':'max','srxMin':'min'})
    dfx=pd.merge(dfx,dft,how='left',on='shop_id')
    dfx['bsRate']=(dfx.bsCount-dfx.bsMin)/(dfx.bsMax-dfx.bsMin)
    dfx['srxRate']=(dfx.srx-dfx.srxMin)/(dfx.srxMax-dfx.srxMin)
    #删除无用行
    dfx=dfx.drop(['srx','bsMax','bsMin','srxMax','srxMin','bsCount'],axis=1)
    #记录序列————wifi强度和强度的Rank
    #去重
    dfy=t2.groupby(['row_id','wifi_id'],as_index=False).head(1)
    dfy['sry']=dfy[['row_id','wifi_id','power']].groupby('row_id').power.rank(ascending=False,method='min')
    dfy.sry=-dfy.sry
    #归一化，平衡记录内部
    dft=dfy.groupby(['row_id'],as_index=False).sry.agg({'sryMax':'max','sryMin':'min'})
    dfy=pd.merge(dfy,dft,how='left',on='row_id')
    dfy['sryRate']=(dfy.sry-dfy.sryMin)/(dfy.sryMax-dfy.sryMin)
    #删除无用行
    dfy=dfy.drop(['sryMax','sryMin','sry'],axis=1)
    
    #合并
    dfz=pd.merge(dfy,dfx,how='left',on='wifi_id')
    del dfx,dfy;gc.collect()
    #如果数据量太大，在此处筛选一波dfz
    #强度差值及归一化，平衡不同bssid-shopid组合强度差的差异
    dfz['ds']=-abs(dfz.power-dfz.bsMedian)
    dft=dfz.groupby(['wifi_id','shop_id'],as_index=False).ds.agg({'dsMax':'max','dsMin':'min'})
    dfz=pd.merge(dfz,dft,how='left',on=['wifi_id','shop_id'])
    dfz['dsRate']=(dfz.ds-dfz.dsMin)/(dfz.dsMax-dfz.dsMin)
    #赋权融合
    i,j,m,n=(0.3,0.3,0,0.6)
    dfz['sr']=i*dfz.srxRate+j*dfz.bsRate+m*dfz.sryRate+n*dfz.dsRate
    dfz=dfz.drop(['dsMax','dsMin'],axis=1)
    #同类求和
    dfz=dfz.groupby(['row_id','shop_id'],as_index=False).sum()
    result=pd.merge(result,dfz[['row_id','shop_id','sryRate','bsRate','srxRate','dsRate','ds','sr']],how='left',on=['row_id','shop_id'])
    cols=['sryRate','bsRate','srxRate','dsRate','ds','sr']
    result,colsAfter=featureInGroup(result,cols)
    return result





    
def get_shop_count_dic(x):
    wificount={}
    wifi=[]
    count=[]
    wifis=x.split('|')
    for i in wifis:
        s=i.split(':')
        wifi.append(s[0])
        count.append(int(s[1]))
    wificount = dict(zip(wifi, count))
    return wificount
def get_wificount(x,y):
    count=0
    try:      
       for i in x:
           if i in y:
              count=count+int(x[i])
           else:
               count=count+0
       return count
    except:
       return 0
def get_wifirat(x, y):
    m=[]
    n=[]
    a={}
    for i in x:
        m.append(i)
        n.append(x[i]/float(y))
    a=dict(zip(m,n))  
    return a   
def wifi_count_intest(result):
    wifi_shop_count['w8']=wifi_shop_count['wifi_id']+':'+wifi_shop_count['wifi inshopcount'].astype('str')
    t1=wifi_shop_count[['shop_id','w8']]
    t2 = t1.groupby('shop_id')['w8'].agg(lambda x:'|'.join(x)).reset_index()
    t2['w8']=t2['w8'].map(get_shop_count_dic)
    del wifi_shop_count['w8']
    result = pd.merge(result,t2,on = ['shop_id'],how = 'left')
    result['wifishopcount'] = map(lambda x, y: get_wificount(x, y) , result['w8'], result['w1'])
    #result['wifishopcountdic'] = map(lambda x, y: get_wifirat(x, y) , result['w8'], result['shop_hot'])
    del result['w8']
    return result
                   
    
def connectamount(result):
    t1=wifi_shop_connect_only[['shop_id','wifi_id']].groupby('shop_id')['wifi_id'].agg(lambda x:':'.join(x)).reset_index()
    t1['h2']=t1['wifi_id'].map(fuck)
    del t1['wifi_id']
    result=pd.merge(result,t1,on=['shop_id'],how='left')
    a=list(result['w1'])
    b=list(result['h2'])
    c=[]
    for i in range(len(a)):
        try:
            d=len(set(a[i])&set(b[i]))
            c.append(d)
        except:
            c.append(0)
    result['h3']=c
    del result['w1']
    del result['h2']
    return result

def choosesamewifi(x):
    b=[]
    from collections import Counter
    f=sorted(Counter(x).iteritems(),key=lambda x:x[1],reverse=True)
    for i in range(10):
        try:
           b.append(f[i][0])
        except:
           return b
    return b
def listsum(x):
    a=[]
    for i in x:
        a=a+i
    return a

def samewifi(result):
    t1=result[['row_id','w2']].groupby('row_id')['w2'].agg(listsum).reset_index()
    t1['wx']=t1['w2'].map(choosesamewifi)
    result=pd.merge(result,t1[['row_id','wx']],on='row_id',how='left')
    
    del result['w2']
    return result
  
def get_wifimostcount(x,y):
    count=0
    try:
        for each in x:
            if each in y:
                count=count+1
        return count
    except:
        return 0
    
def get_mostcountin(x,y):
    try:
        for each in x:
            if each in y:
               return 1
            else:
               return 0
    except:
        return 0    
def get_tolist(x):
    a=[]
    wifi=x.split('|')
    for each in wifi:
        a.append(each)
    return a

def wifishopmaxin(train,result):
    t1= train[['shop_id','wifi_infos']].drop_duplicates()
    t2=make_wifi_shop_relation(t1)
    t5=t2.copy()
    t2=t2.groupby(['shop_id','wifi_id'],as_index=False)['power'].agg({'max_power':'max'}) 
    t3=t2.groupby('wifi_id',as_index=False)['max_power'].agg({'max_power':'max'})   
    t4=pd.merge(t3,t2,on=['max_power','wifi_id'],how='left')
    t4['mostwifi']=t4['wifi_id']+':'+t4['max_power'].astype('str')
    t4=t4[['shop_id','mostwifi']]
    t4= t4.groupby('shop_id')['mostwifi'].agg(lambda x:'|'.join(x)).reset_index()
    t4['mostwifi']=t4['mostwifi'].map(get_shop_wifi_dic)
    result = pd.merge(result,t4,on = ['shop_id'],how = 'left')
    result['ww']=result['wifi_infos'].map(get_wifi_power)
    result['wifimostcount']=map(lambda x, y: get_wifimostcount(x, y) , result['ww'], result['mostwifi'])
    
    t5=t5.groupby(['shop_id','wifi_id'],as_index=False)['power'].agg({'countall':'count'}) 
    t6=t5.groupby('shop_id',as_index=False)['countall'].agg({'countall':'max'})
    t6=pd.merge(t6,t5,on=['shop_id','countall'],how='left')
    t6=t6.groupby('shop_id')['wifi_id'].agg(lambda x:'|'.join(x)).reset_index()
    t6['wifi_id']=t6['wifi_id'].map(get_tolist)
    t6.rename(columns={'wifi_id':'wifi_need'},inplace=True)
    result = pd.merge(result,t6[['wifi_need','shop_id']],on = ['shop_id'],how = 'left')
    result['mostcountin']=map(lambda x, y: get_mostcountin(x, y) , result['ww'], result['wifi_need'])
    
    t7=t2.groupby(['shop_id'],as_index=False)['max_power'].agg({'max_power':'max'})
    t8=pd.merge(t7,t2,on=['max_power','shop_id'],how='left') 
    t8=t8.groupby('shop_id')['wifi_id'].agg(lambda x:'|'.join(x)).reset_index()
    t8['wifi_id']=t8['wifi_id'].map(get_tolist)
    t8.rename(columns={'wifi_id':'wifi1_need'},inplace=True)    
    result = pd.merge(result,t8,on = ['shop_id'],how = 'left')
    result['mostpowerin']=map(lambda x, y: get_mostcountin(x, y) , result['ww'], result['wifi1_need'])
    
    
    
    
    
    
    
    del result['ww']
    
    return result




    
def get_wifi_power(x):
    wifipower={}
    wifi=[]
    power=[]
    wifis=x.split(';')
    for i in wifis:
        s=i.split('|')
        wifi.append(s[0])
        power.append(int(s[1]))
    wifipower = dict(zip(wifi, power))
    return wifipower

def get_shop_wifi_dic(x):
    wifipower={}
    wifi=[]
    power=[]
    wifis=x.split('|')
    for i in wifis:
        s=i.split(':')
        wifi.append(s[0])
        power.append(float(s[1]))
    wifipower = dict(zip(wifi, power))
    return wifipower

def get_knn(x,y):
    sum=0
    for each in x:
        try:
            if each in y:
               sum+=(x[each]-y[each])**2    #历史信息里存在这个信号的话，计算
            else:
               sum += (x[each]-(-90)) ** 2
        except:
            sum=999        
    return sum**0.5*1.0/len(x)


def get_std(x,y):
    a=[]
    try:
       for each in x:
           if each in y:
               b=abs(x[each]-y[each])
               a.append(b)
       return np.std(a)
    except:
       return 999
def get_mean(x,y):
    a=[]
    try:
       for each in x:
           if each in y:
               b=abs(x[each]-y[each])
               a.append(b)
       return np.mean(a)
    except:
       return 999

def get_weightpower(x, y):
    sum=0
    for each in x:
        if each in y:
            try:
               sum=sum+float((120+x[each])*(y[each]))
            except:
                sum=sum+0
        else:
            sum=sum+0
    return sum

def find_larger_avp(x,y):
    count=0
    for each in x:
        if each in y:
           if x[each]>=y[each]:
              count=count+1
    return count

def find_short_min(x, y):
    count=0
    for each in x:
        if each in y:
           if x[each]<=y[each]:
              count=count+1
        else:
            count=count+1
    return count    
def find_bigger_max(x,y):
    count=0
    for each in x:
        if each in y:
           if x[each]>=y[each]:
              count=count+1
    return count 
def find_bigger_ratio(x,y):
    count=0
    sum=0
    for each in x:
        if each in y:
           sum=sum+1
           if x[each]>=y[each]:
              count=count+1
    if sum==0:
        return 0
    else:
        return count/float(sum) 
def get_weighted_knn(x,y,z):
    sum=0
    try:
       for each in x:
           if each in y:
              if each in z:
                 sum=sum+(1-z[each])*(abs(x[each]-y[each]))
           else:
               sum=sum+abs(x[each]-(-90))
    except:
        sum=999
    return sum

def get_knn1(x,y,z):
    sum=0
    count=0
    try:
       for each in x:
           if each in z:            
               count=count+1
               sum+=(x[each]-y[each])**2    #历史信息里存在这个信号的话，计算
       if count==0:
          return 99
       else:    
          return sum**0.5*1.0/count
    except:
          return 99

def get_knn2(x,y,z):
    sum=0
    count=0
    try:
       for each in x:
            if each in z:
               count+=1
               if each in y:
                   sum+=abs(x[each]-y[each])
               else:
                   sum+=abs(x[each]-(-90))
            
       if count==0:
           return 99 
       else: 
           return sum
    except:
         return 999
def get_sametopwifi(x,y,z):
    count=0
    try:
       for each in x:
           if each in z:
              if each in y:
                 count+=1
           else:
               count+=0
       return count
    except:
         return 0
def find_smallest(x,y):
    power=[]
    try:
        for each in x:
            if each in y:
               power.append(x[each])
        return min(power)
    except:
        return -100

def get_cos(x,y):
    power1=0
    power2=0
    power3=0
    try:
       for each in x:
           if each in y:
              m=x[each]*y[each]
              power1=power1+m
              n=x[each]**2
              power2=power2+n
              z=y[each]**2
              power3=power3+z
       s=float(power1)/((power2**0.5)*(power3**0.5))
       return s
    except:
        return -1
def get_shop_wifi_alldic(x):
    wifipower={}
    wifis=x.split('|')
    for i in wifis:
        s=i.split(':')
        wifipower.setdefault(s[0],[]).append(int(s[1]))
    return wifipower
def get_infive(x,y):
    count=0
    try:
       for each in x:
           if each in y:
              for i in y[each]:
                  if x[each]>=i:
                      count=count+1
       return count
    except:
        return 0  
def get_better(x,y):
    count1=0
    count=0
    sum=0
    try:
       for each in x:
           if each in y:
              for i in y[each]:
                  count1=count1+1
                  if x[each]>=i:
                      count=count+1
              sum=sum+float(count)/count1
       return sum
    except:
        return 0    
    
    
def get_wifidistance(x,y):
    sum=0
    count=0
    for each in x:
        if each in tx:
           count=count+1
           if each in y:
               a=abs(tx[each]-y[each])
           else:
               a=abs(tx[each]-(-90))
        else:
            a=0
        sum=sum+a
    if count==0:
        return 99
    else:
        return sum/float(count)
def get_max_wifi_dis(x):
    sum=0
    count=0
    for each in x:
        if each in tx:
           count=count+1
           a=abs(tx[each]-x[each])
        else:
           a=0
        sum=sum+a
    if count==0:
        return 20
    else:
        return sum/float(count)   


def get_max_wifi_dismin(x):
    sum=0
    count=0
    for each in x:
        if each in tx_min:
           count=count+1
           a=abs(tx_min[each]-x[each])
        else:
           a=0
        sum=sum+a
    if count==0:
        return 20
    else:
        return sum/float(count)  



     
    
def ger_power_var(train,result):
    result['w4']=result['wifi_infos'].map(get_wifi_power)
    wifi_shop_info['w']=wifi_shop_info['wifi_id']+':'+wifi_shop_info['ave_power'].astype('str')
    t1=wifi_shop_info[['shop_id','w']]
    t2 = t1.groupby('shop_id')['w'].agg(lambda x:'|'.join(x)).reset_index()
    t2['w']=t2['w'].map(get_shop_wifi_dic)    
    result = pd.merge(result,t2,on = ['shop_id'],how = 'left')
    result['knn_values'] = map(lambda x, y: get_knn(x, y) , result['w4'], result['w'])
    result['knn_std'] = map(lambda x, y: get_std(x, y) , result['w4'], result['w'])
    result['knn_mean']=map(lambda x, y: get_mean(x, y) , result['w4'], result['w'])
    result['avgcossimilarity'] = map(lambda x, y: get_cos(x, y) , result['w4'], result['w'])
    result['larger1'] = map(lambda x, y:find_larger_avp(x, y) , result['w4'], result['w'])
    result['larger1_ratio'] = map(lambda x, y:find_bigger_ratio(x, y) , result['w4'], result['w'])
    result['powersmallest']=map(lambda x, y: find_smallest(x, y) , result['w4'], result['w'])
    result['short2'] = map(lambda x, y:find_short_min(x, y) , result['w4'], result['w'])
    result['rua1'] = map(lambda x,y,z: get_weighted_knn(x,y,z) , result['w4'], result['w'],result['s'])
    result['rua3'] = map(lambda x,y,z: get_knn1(x,y,z) , result['w4'], result['w'],result['p'])   
    result['rua5'] = map(lambda x,y,z: get_knn2(x,y,z) , result['w4'], result['w'],result['wx'])   
    result['rua6'] = map(lambda x,y,z: get_sametopwifi(x,y,z) , result['w4'], result['w'],result['wx'])          
    #result['x1'] = map(lambda x, y: get_weightpower(x, y) , result['w4'], result['s'])
    #result['x2']=result['x1']/result['w3'].astype('float')
    
    wifi_shop_min['n1']=wifi_shop_min['wifi_id']+':'+wifi_shop_min['min_power'].astype('str')
    t1=wifi_shop_min[['shop_id','n1']]
    t2 = t1.groupby('shop_id')['n1'].agg(lambda x:'|'.join(x)).reset_index()
    t2['n1']=t2['n1'].map(get_shop_wifi_dic)    
    result = pd.merge(result,t2,on = ['shop_id'],how = 'left')
    result['short1']=map(lambda x, y:find_short_min(x, y),result['w4'], result['n1'])
    result['knn_min_values'] = map(lambda x, y: get_knn(x, y) , result['w4'], result['n1'])
    result['short3']=map(lambda x, y:find_bigger_max(x, y),result['w4'], result['n1'])  
    result['short3_ratio']=map(lambda x, y:find_bigger_ratio(x, y),result['w4'], result['n1']) 
    
    
    
    
    wifi_shop_max['n']=wifi_shop_max['wifi_id']+':'+wifi_shop_max['max_power'].astype('str')
    t1=wifi_shop_max[['shop_id','n']]
    t2 = t1.groupby('shop_id')['n'].agg(lambda x:'|'.join(x)).reset_index()
    t2['n']=t2['n'].map(get_shop_wifi_dic)    
    result = pd.merge(result,t2,on = ['shop_id'],how = 'left')
    result['knn_max_values'] = map(lambda x, y: get_knn(x, y) , result['w4'], result['n'])
    result['maxcossimilarity'] = map(lambda x, y: get_cos(x, y) , result['w4'], result['n'])
    result['rua2'] = map(lambda x,y,z: get_weighted_knn(x,y,z) , result['w4'], result['n'],result['s'])    
    result['larger2'] = map(lambda x, y:find_bigger_max(x, y) , result['w4'], result['n'])
    result['rua4'] = map(lambda x,y,z: get_knn1(x,y,z) , result['w4'], result['n'],result['p'])   
    result['rua7']=map(lambda x,y,z: get_knn2(x,y,z) , result['w4'], result['n'],result['wx'])   
    
    
    
    t3= train[['shop_id','wifi_infos','row_id']].drop_duplicates()
    t3=f_owen(t3)
    t3['allwifipower']=t3['wifi_id']+':'+t3['power'].astype('str')
    t3=t3[['shop_id','allwifipower']]
    t4= t3.groupby('shop_id')['allwifipower'].agg(lambda x:'|'.join(x)).reset_index()
    t4['allwifipower']=t4['allwifipower'].map(get_shop_wifi_alldic)    
    result = pd.merge(result,t4,on = ['shop_id'],how = 'left')   
    result['rua8']=map(lambda x,y: get_infive(x,y) , result['w4'], result['allwifipower'])  
    result['rua9']=map(lambda x,y: get_better(x,y) , result['w4'], result['allwifipower'])  
    
    
    
    global tx
    t5= train[['shop_id','wifi_infos']].drop_duplicates()
    t6=make_wifi_shop_relation(t5)
    t6=t6.groupby('wifi_id',as_index=False)['power'].agg({'max_power':'max'})   
    tx=dict(zip(t6.wifi_id,t6.max_power))
    result['rua10']=map(lambda x,y: get_wifidistance(x,y) , result['w4'], result['n'])      
    result['rua11']=result['w4'].map(get_max_wifi_dis)
    
    
    global tx_min
    t7=t6.groupby('wifi_id',as_index=False)['power'].agg({'min_power':'min'})   
    tx=dict(zip(t7.wifi_id,t7.max_power))
    result['rua12']=result['w4'].map(get_max_wifi_dismin)
    
    
    
    
    
    #del result['w4']
    del result['w']
    del result['n']
   # del result['n1']
    del result['s']
    del result['p']
    del result['allwifipower']
    return result
    

#shop_id和当前信号最强的wifi以及第二强的wifi有没有互联记录
def if_wifi_shop_connected(result):
    global wifi_shop_connect_only
    wifi_shop_connect_only.rename(columns={'wifi_id':'wifi1_id','connect':'connect1'},inplace=True)
    result=pd.merge(result,wifi_shop_connect_only,on=['wifi1_id','shop_id'],how='left')
    wifi_shop_connect_only.rename(columns={'wifi1_id':'wifi2_id','connect1':'connect2'},inplace=True)
    result=pd.merge(result,wifi_shop_connect_only,on=['wifi2_id','shop_id'],how='left')
    wifi_shop_connect_only.rename(columns={'wifi2_id':'wifi_id','connect2':'connect'},inplace=True)
    result['connect1']=result['connect1'].fillna(0)
    result['connect2']=result['connect2'].fillna(0)
    result['connect_sum']=result['connect1']+result['connect2']
    return result
#找一下距离差
#加上商店历史上的平均距离，商店位置表中的距离不准
def get_real_shop_loc(result):
    global shop_loc
    shop_loc=shop_loc.groupby('shop_id',as_index=False).agg({'longitude':'mean','latitude':'mean'})
    shop_loc.rename(columns={'longitude':'lon_real','latitude':'lat_real'},inplace=True)
    result=pd.merge(result,shop_loc,on='shop_id',how='left')
    shop_loc.rename(columns={'lon_real':'longitude','lat_real':'latitude'},inplace=True)
    return result

def distance_real_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.lon_real,result.lat_real))
    sum=list(zip(loc1,loc2))
    dis_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_distance(lat1, lon1,lat2, lon2)
        dis_df1.append(d)
    result['real_dis_df']=dis_df1
    return result
def angle_real_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.lon_real,result.lat_real))
    sum=list(zip(loc1,loc2))
    ang_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_orientation(lat1, lon1,lat2, lon2)
        ang_df1.append(d)
    result['real_ang_df']=ang_df1
    return result


def distance_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.longitude_s,result.latitude_s))
    sum=list(zip(loc1,loc2))
    dis_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_distance(lat1, lon1,lat2, lon2)
        dis_df1.append(d)
    result['dis_df']=dis_df1
    return result
#找一下角度差
def angle_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.longitude_s,result.latitude_s))
    sum=list(zip(loc1,loc2))
    ang_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_orientation(lat1, lon1,lat2, lon2)
        ang_df1.append(d)
    result['ang_df']=ang_df1
    result['wifi_count_rt']=result['wifishopcount'].astype('float')/result['shop_hot'] 
    result['top_count_rt']=result['topcount'].astype('float')/result['shop_hot'] 
    result['rua8ratio']=result['rua8'].astype('float')/result['shop_hot']
    cols=['wifiprosum','knn_max_values','scorewifi','wifi_count_rt','knn_std']
    result,colsAfter=featureInGroup(result,cols)     
    return result

#行为发生时间
def get_time(train,result):
    result=get_time_hour(result)
    train=get_time_hour(train)
    t1=train.groupby('shop_id',as_index=False)['hourofday'].agg({'h_s_m':'mean','h_s_v':'std','h_s':'median'})
    t2=train.groupby(['shop_id','hourofday'],as_index=False)['row_id'].agg({'shophourcount':'count'})
    result=pd.merge(result,t1,on='shop_id',how='left')
    result=pd.merge(result,t2,on=['shop_id','hourofday'],how='left')
    result['time_dif']=abs(result['h_s']-result['hourofday'])
    return result


# 构造候选样本
def get_sample(train,test):
    result_path = cache_path + 'sample_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        result1= get_wifi_shop(train, test)            # 根据wifi历史来添加样本
        result2=GetGeoCandidate(train,test)
        result3=GetLCSCandidate(train,test)
        result4=get_user_end_shop(train,test)
        result = pd.concat([result1,result2,result3,result4]).drop_duplicates()
        test_temp = test.copy()
        result = pd.merge(result, test_temp, on='row_id', how='left')
        del result['mall_id']
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 制作训练集
def make_train_set(train,test):
    global wifi_shop_count
    global wifi_shop_info
    global wifi_shop_connect_only
    global wifi_shop_max
    global wifi_shop_min
    print('make wifishop info')
    wifi_shop_connect1=wifi_shop_connect(train)
    wifi_shop_connect_only=wifi_shop_connect1.drop_duplicates()

    wifi_shop_info= get_wifi_shop_info(train)
    wifi_shop_max=get_wifi_shop_max(train)
    wifi_shop_min=get_wifi_shop_min(train)
    wifi_shop_count= get_wifi_shop_count(train)
    print('make sample...')
    result = get_sample(train,test)                                         
    del test['mall_id']
    train=pd.merge(train,shop,on=['shop_id'],how='left')
    
    print('make character...')
    
    result = get_user_count(train,result)                                   
    result=get_shop_hot(train,result)
    result = get_user_shop_count(train, result)                             
    result =get_shop_info(result)                                    
    result = get_uesr_kind(train, result)                  
    result = get_wifi(result,train) 
    result=get_same_relation(result)
    result=get_top10_sameresult(train,result)
    result=shop_pro(train,result)
    result=get_shop_score(train,result)
    result=featureLCS(train,test,result)
    result= wifi_count_intest(result)
    result=connectamount(result)
    result=samewifi(result)
    result=wifishopmaxin(train,result)
    result=ger_power_var(train,result)
    result=if_wifi_shop_connected(result)  
    result=get_time(train,result) 
    result=get_real_shop_loc(result)
    result=distance_real_dif(result)
    result=angle_real_dif(result)
    result=distance_dif(result)
    result=angle_dif(result)
    result['user_id']=result.user_id.map(lambda x:int(str(x)[2:]))       
    result['mall_id']=result.mall_id.map(lambda x:int(str(x)[2:]))     
    result['real_lat_dif']=abs(result['lat_real']-result['latitude'])
    result['real_lon_dif']=abs(result['lon_real']-result['longitude']) 
   # result['wifi_count_rt']=result['wifishopcount'].astype('float')/result['shop_hot']       
    result.fillna(0,inplace=True)
    print('result.columns:\n{}'.format(result.columns))   
    
    return result
    


# 训练提交

if __name__ == "__main__":
    t0 = time.time()
    train = pd.read_csv(train_path)
    shop=pd.read_csv(shop_path)
    shop_loc=train[['shop_id','longitude','latitude']]
    shop.rename(columns={'longitude':'longitude_s'},inplace=True)
    shop.rename(columns={'latitude':'latitude_s'},inplace=True)
    train['row_id']=range(train.shape[0])
    train['row_id']='t'+train['row_id'].astype('str')
    shop1=shop[['shop_id','mall_id']]    
    train1 = train[(train['time_stamp']< '2017-08-18 00:00:00')]
    train2 = train[(train['time_stamp']>= '2017-08-18 00:00:00')&(train['time_stamp']< '2017-08-25 00:00:00')]
    test=train[(train['time_stamp']>= '2017-08-25 00:00:00')]
    train=pd.concat([train1,train2])
    train2=pd.merge(train2,shop1,on=['shop_id'],how='left')

    test=pd.merge(test,shop1,on=['shop_id'],how='left')
    real=test[['row_id','shop_id']]
    del train2['shop_id']
    del test['shop_id']
    print('make train_feat')
    
    
    train_feat = make_train_set(train1,train2) #生成变量，这里的train1当做训练集，train2当做测试集，提取特征
    print('add label')
    train_feat= get_label(train_feat) #根据候选集生成binary-label
    print('finish add_label')
    print ('pos/all=%f' %(len(train_feat[train_feat['label']==1])*1.0/(train2.shape[0])))
    
    print('make test_feat')
    
    test_feat = make_train_set(train,test) #提取特征
    import xgboost as xgb
    predictors = ['larger1_ratio','short3_ratio','short3','knn_min_values','rua12','rua11','rua10','mostpowerin','mostcountin','wifimostcount','knn_std_groupRate','knn_std','knn_mean','rua9','rua8ratio','rua8','shophourcount','top_count_rt','topcount','avgcossimilarity','maxcossimilarity','powersmallest','wifiprosum_groupRate',
    'knn_max_values_groupRate','wifi_count_rt_groupRate','scorewifi_groupRate','rua5','rua6','rua7','sryRate_groupRate',
    'bsRate_groupRate','srxRate_groupRate','dsRate_groupRate','ds_groupRate','sr_groupRate','sryRate','bsRate','srxRate',
    'dsRate','ds','sr','wifitopcount','rua4','rua3','wifiinrat','wifinoshow','wifipromulti','rua2','short2','rua1','wifiturerat',
    'larger2','larger1','short1','scorewifi','wifiprosum','h_s','time_dif','h_s_v','h_s_m','knn_max_values','wifi_count_rt',
    'wifishopcount','h3','knn_values','real_lon_dif','real_lat_dif','w3','lon_real','lat_real','real_ang_df','real_dis_df','connect_sum',
    'ang_df','dis_df','mall_id','user_id','dayofweek','hourofday','connect1','connect2','power_dif2','ave_power2','power_dif1',
    'ave_power1','wifi2_power','wifi1_power','user_kind_count','category','user_shop_count','latitude_s','longitude_s',
   'shop_hot','user_count']
   
    params = {
        'objective': 'binary:logistic',
        'eta': 0.08,
        'colsample_bytree': 0.886,
        'min_child_weight': 1.1,
        'max_depth': 7,
        'subsample': 0.886,
        'gamma': 0.1,
        'lambda':10,
        'verbose_eval': True,
        'eval_metric': 'auc',
        'scale_pos_weight':6,
        'seed': 201703,
        'missing':-1
   }

    xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['label'])
    xgbtest = xgb.DMatrix(test_feat[predictors])
    print ('start training')
    model = xgb.train(params, xgbtrain, num_boost_round=500)
    gc.collect()
    print ('start predicting')
    test_feat.loc[:,'pred'] = model.predict(xgbtest)
    test_feat1 = test_feat[['row_id','shop_id','pred']].drop_duplicates()
    gc.collect()
    result=test_feat1.groupby('row_id',as_index=False)['pred'].agg({'pred':'max'})
    result=pd.merge(result,test_feat1,on=['row_id','pred'],how='left')
    result = pd.merge(test[['row_id']],result[['row_id','shop_id']],on='row_id',how='left')
    result.fillna('0',inplace=True)
    real.rename(columns={'shop_id':'real_shopid'},inplace=True)
    score=pd.merge(real,result,on='row_id',how='left')
    score['lebel']=(score['real_shopid'] == score['shop_id']).astype('int')
    accuracy=score[score['lebel']==1].shape[0]/float(test.shape[0])
    print accuracy
    print('一共用时{}秒'.format(time.time()-t0))
    xgb.plot_importance(model)
    
    
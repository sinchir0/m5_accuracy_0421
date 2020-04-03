import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import gc
import os
        
import sys
sys.path.append('.')
from configs.local_parameter import *

# # 欠損値を埋める、LabelEncoder実施
# def transform(data):
    
#     nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
#     for feature in nan_features:
#         data[feature].fillna('unknown', inplace = True)
        
#     cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
#     for feature in cat:
#         encoder = preprocessing.LabelEncoder()
#         data[feature] = encoder.fit_transform(data[feature])
    
#     return data

# data = transform(data)
if USE_ALL_DATA:
    data = pd.read_pickle(f"{DATA_PATH}/data_all.pkl")
else:
    data = pd.read_pickle(f"{DATA_PATH}/data.pkl")

def simple_fe(data):
    
    to_pickle_fe_list = []
    
    #### lag demand feature
    # shift = 28行下にデータを移す、つまり28日前のデータといういこと
    data['001_lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))    
    data['001_lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))    
    data['001_lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))  
    
    #### rolling demand features
    # 7点について、meanだったり、stdだったりを取る。
    data['002_rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())    
    data['002_rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    data['002_rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    data['002_rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    data['002_rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    data['002_rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    data['002_rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
    data['002_rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())
    
    to_pickle_fe_list.extend([
        '001_lag_t28',
        '001_lag_t29',
        '001_lag_t30',
        '002_rolling_mean_t7',
        '002_rolling_std_t7',
        '002_rolling_mean_t30',
        '002_rolling_mean_t90',
        '002_rolling_mean_t180',
        '002_rolling_std_t30',
        '002_rolling_skew_t30',
        '002_rolling_kurt_t30'
    ])
    
    #### price feature
    data['003_lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))    
    data['003_price_change_t1'] = (data['003_lag_price_t1'] - data['sell_price']) / (data['003_lag_price_t1'])    
    data['003_rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['003_price_change_t365'] = (data['003_rolling_price_max_t365'] - data['sell_price']) / (data['003_rolling_price_max_t365'])
    data['003_rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['003_rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    #data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
    
    to_pickle_fe_list.extend([
        '003_lag_price_t1',
        '003_price_change_t1',
        '003_rolling_price_max_t365',
        '003_price_change_t365',
        '003_rolling_price_std_t7',
        '003_rolling_price_std_t30',
    ])
    
    #### time feature
    data['date'] = pd.to_datetime(data['date'])
    data['004_year'] = data['date'].dt.year
    data['004_month'] = data['date'].dt.month
    data['004_week'] = data['date'].dt.week
    data['004_day'] = data['date'].dt.day
    data['004_dayofweek'] = data['date'].dt.dayofweek
    
    to_pickle_fe_list.extend([
        '004_year',
        '004_month',
        '004_week',
        '004_day',
        '004_dayofweek',
    ])
    
    # to_pickle
    def series_to_df_to_pickle(df,fe_name):
        pd.DataFrame(df[fe_name]).to_pickle(f'{FEATURES_PATH}/{fe_name}.pkl')
    
    for fe_name in to_pickle_fe_list:
        series_to_df_to_pickle(data,fe_name)
    
    #return data
    
simple_fe(data)
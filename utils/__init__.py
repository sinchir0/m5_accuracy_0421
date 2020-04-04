import pandas as pd
import numpy as np
import os, sys, gc, warnings, random, datetime
import argparse
import json
import pickle
from configs.local_parameter import *
import gc

# auto_label_encoder
import category_encoders as ce

def load_datasets(feats):
    #dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    dfs = [pd.read_pickle(f'{ROOT_PATH}/features/{f}_train.pkl') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    #dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    dfs = [pd.read_pickle(f'{ROOT_PATH}/features/{f}_test.pkl') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test

def load_target(target_name):
    train = pd.read_csv(f'{ROOT_PATH}/data/input/train.csv')
    y_train = train[target_name]
    return y_train

def load_datasets_and_target(feats,target_name):
    #dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    #dfs = [pd.read_pickle(f'{ROOT_PATH}/features/{f}_train.pkl') for f in feats]
    #X_train = pd.concat(dfs, axis=1, sort=False)
    #dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    #dfs = [pd.read_pickle(f'{ROOT_PATH}/features/{f}_test.pkl') for f in feats]
    #test_df = pd.concat(dfs, axis=1, sort=False)
    
    # 特徴量の読み込み
    features = [pd.read_pickle(f'{ROOT_PATH}/features/{f}.pkl') for f in feats]
    features_concat = pd.concat(features, axis=1, sort=False)
    
    # ベースデータの読み込み
    if USE_ALL_DATA:
        data_not_concat = pd.read_pickle(f"{DATA_PATH}/data_all.pkl")
    else:
        data_not_concat = pd.read_pickle(f"{DATA_PATH}/data.pkl")
    
    #　ベースデータと特徴量の結合
    data = pd.concat([data_not_concat,features_concat], axis=1, sort=False)
    
    # メモリの節約
    data = reduce_mem_usage(data)
    
    del data_not_concat,features_concat
    gc.collect()
    
    if AUTO_LABEL_ENCODER_IN_LOAD_DATASETS_AND_TARGET:
        data = label_encoder(data)
    
    # InfをNaNに変換
    if INF_TO_NAN:
        data = inf_to_nan(data)
    
    # train,testへの分割
    train_df = data[data['date'] <= '2016-03-27']
    X_train = train_df.drop(TARGET, axis = 1)
    y_train = train_df[TARGET]
    
    valid_df = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    X_valid = valid_df.drop(TARGET, axis = 1)
    y_valid = valid_df[TARGET]
    
    test_df = data[(data['date'] > '2016-04-24')]
    
    if OPE_CHECK:
        train_df = train_df[1:1000]; X_train = X_train[1:1000]; y_train = y_train[1:1000]
        valid_df = valid_df[1:1000]; X_valid = X_valid[1:1000]; y_valid = y_valid[1:1000]
    
    del data
    gc.collect()
    
    return train_df,valid_df,test_df,X_train,y_train,X_valid,y_valid

#def seed_everything(seed=71):
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# def config_read():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', default='./configs/default.json')
#     options = parser.parse_args()
#     config = json.load(open(options.config))
#     return config

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def label_encoder(all_data):    
    # Labelencを行うリストとして、dtype=objectを取得
    obj_list = all_data.select_dtypes("O").columns.tolist() 
    
    # ラベルエンコーディングしたくない列を除外する
    for f in EXCLUDE_COLUMNS:
        obj_list.remove(f)

    # 文字を序数に変換
    ce_oe = ce.OrdinalEncoder(cols=obj_list)
    all_data_after_le_only_obj_col = ce_oe.fit_transform(all_data)
    
    # 公式の推奨である０始まりに一応する
    for feature in obj_list:
        all_data[feature] = all_data_after_le_only_obj_col[feature] - 1

    return all_data

def auto_label_encoder(train,valid,test):    
    # Labelencを行うリストとして、dtype=objectを取得
    obj_list = train.select_dtypes("O").columns.tolist() 
    
    # ラベルエンコーディングしたくない列を除外する
    for f in EXCLUDE_COLUMNS:
        obj_list.remove(f)
    
    # 区分のためにtypeを記載
    train["type"] = "train"
    valid["type"] = "valid"
    test["type"] = "test"
    
    # データを全結合
    all_data = pd.concat([train,valid,test],sort=False)
    all_data_after_le = all_data.copy()

    # 文字を序数に変換
    ce_oe = ce.OrdinalEncoder(cols=obj_list)
    all_data_after_le_only_obj_col = ce_oe.fit_transform(all_data)
    
    # 公式の推奨である０始まりに一応する
    for feature in obj_list:
        all_data_after_le[feature] = all_data_after_le_only_obj_col[feature] - 1
    
    # trainとtestに分割
    train_after = all_data_after_le[all_data_after_le["type"] == "train"].drop('type', axis = 1)
    valid_after = all_data_after_le[all_data_after_le["type"] == "valid"].drop('type', axis = 1)
    test_after = all_data_after_le[all_data_after_le["type"] == "test"].drop('type', axis = 1)

    return train_after, valid_after, test_after

def inf_to_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)
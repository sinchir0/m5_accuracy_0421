import pandas as pd
import numpy as np

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

import category_encoders as ce

def auto_label_encoder(train,valid,test):    
    # Labelencを行うリストを取得
    obj_list = train.select_dtypes("O").columns.tolist() 
    
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
    train_after = all_data_after_le[all_data_after_le["type"] == "train"]
    valid_after = all_data_after_le[all_data_after_le["type"] == "valid"]
    test_after = all_data_after_le[all_data_after_le["type"] == "test"]

    return train_after, valid_after, test_after
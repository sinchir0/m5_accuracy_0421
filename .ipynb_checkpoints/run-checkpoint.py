# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging

from models.Base_Model import *
from models.lgbm import *

from configs.local_parameter import *
from utils import *

seed_everything()

print(f"Start {CASE}")

if OPE_CHECK:
    print("OPE_CHECK True")    
    
# 一番最初に、今回のケースのディレクトリを作成する。
if not os.path.exists(f'{MODEL_PASS}/{CASE}'):
    os.mkdir(f'{MODEL_PASS}/{CASE}')
    print(f'Make directory : {MODEL_PASS}/{CASE}')

#loggngの設定
os.makedirs(f"./{MAKE_PATH}",exist_ok=True)
#メインの設定
logging.basicConfig(
    filename=f'{MAKE_PATH}/log_{CASE}_{NOW:%Y%m%d%H%M%S}.log', level=logging.DEBUG
)
print("start make log")
#一行目
logging.debug(f'{MAKE_PATH}/log_{CASE}_{NOW:%Y%m%d%H%M%S}.log')

# モデルの読み込み
print("Start load_datasets_and_target")
#train_df,valid_df,test_df,X_train,y_train,X_valid,y_valid = load_datasets_and_target(FEATURE,TARGET)
train_df,valid_df,test_df = load_datasets_and_target(FEATURE,TARGET)
print("Finish")

# その他、出力したいログ群
logging.debug(f"feats:{FEATURE}")
logging.debug(f"target:{TARGET}")
logging.debug(f"train_df.shape:{train_df.shape}")
logging.debug(f"train_df.shape:{valid_df.shape}")
logging.debug(f"test_df.shape:{test_df.shape}")

logger = logging.getLogger('main')

print(train_df)
print(valid_df)
print(test_df)

print("Start lgb_model")
lgb_model = Lgb_Model(train_df = train_df, 
                      test_df = test_df,
                      valid_df = valid_df
                      #n_splits = FOLD, 
                      #categoricals=categoricals
                     )
print("Finish")

# submitファイルの作成
print("Start make submission")

def transform_predict(test, submission):
    test["demand"] = lgb_model.y_pred
    predictions = test[['id', 'date', 'demand']]
    print(predictions)
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    print(predictions)
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    #final.to_csv('submission.csv', index = False)
    return final
    
submission = pd.read_pickle(f"{DATA_PATH}/sample_submission.pkl")

final = transform_predict(test_df, submission)

final.to_csv(f'./submission/sub_{CASE}_{NOW:%Y%m%d%H%M%S}_{lgb_model.score}.csv',
   index=False
)

print(f"Finish {CASE}! I kept you waiting !")
print("Quick submit command")
print(f"python submit_m5_acc.py sub_{CASE}_{NOW:%Y%m%d%H%M%S}_{lgb_model.score}.csv comment")
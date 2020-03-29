# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging

from models.Base_Model import *
from models.lgbm import *

from configs.local_parameter import *
from utils.util_code import *

seed_everything()

print("start load_datasets_and_target")
train_df,valid_df,test_df,X_train,y_train,X_valid,y_valid = load_datasets_and_target(FEATURE,TARGET)
print("finish")


#loggngの設定
os.makedirs(f"./logs/{CASE}",exist_ok=True)
#メインの設定
logging.basicConfig(
    filename=f'./logs/{CASE}/log_{CASE}_{NOW:%Y%m%d%H%M%S}.log', level=logging.DEBUG
)
#一行目
logging.debug(f'./logs/log_{CASE}_{NOW:%Y%m%d%H%M%S}.log')
#その他、出力したいログ群
logging.debug(f"feats:{FEATURE}")
logging.debug(f"target:{TARGET}")
logging.debug(f"train_df.shape:{train_df.shape}")
logging.debug(f"test_df.shape:{test_df.shape}")

logger = logging.getLogger('main')

print(train_df)
print(valid_df)
print(test_df)

print("start lgb_model")
lgb_model = Lgb_Model(train_df = train_df, 
                      test_df = test_df,
                      valid_df = valid_df
                      #n_splits = FOLD, 
                      #categoricals=categoricals
                     )
print("finish")

# submitファイルの作成
print("start make submission")

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

print("finish all! I kept you waiting !")
# CASE
CASE = "case1"

# 動作確認のみ
#OPE_CHECK =True
OPE_CHECK =False

# PATH
ROOT_PATH = "/home/jupyter/m5_accuracy/kaggle-pipeline2"
DATA_PATH = f"{ROOT_PATH}/data/input"
FEATURES_PATH = f"{ROOT_PATH}/features"
MODEL_PASS = f"{ROOT_PATH}/model_output"
SUB_PASS = f"{ROOT_PATH}/submission"

MAKE_PATH = f"{MODEL_PASS}/{CASE}"
PATH_W = f"{MODEL_PASS}/{CASE}/{CASE}_lgb_score.txt"

# SEED
SEED = 2020
#SEED = 71

# USE FEATURE
FEATURE = [
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
    '002_rolling_kurt_t30',
    #'003_lag_price_t1',
    '003_price_change_t1',
    #'003_rolling_price_max_t365',
    '003_price_change_t365',
    '003_rolling_price_std_t7',
    '003_rolling_price_std_t30',
    '004_year',
    '004_month',
    '004_week',
    '004_day',
    '004_dayofweek',
]

USE_ORG_FEATURE = [
    #'id',
    'item_id',
    'dept_id',
    'cat_id',
    'store_id', 
    'state_id', 
    #'demand', this is target feature
    'part', 
    #'date', 
    'wm_yr_wk', 
    'event_name_1', 
    'event_type_1',
    'event_name_2', 
    'event_type_2', 
    'snap_CA', 
    'snap_TX', 
    'snap_WI',
    'sell_price'
]

# to_pickleの対象のCSVの名称を記載
TO_PICKLE_TARGET = [
    "calendar",
    "sales_train_validation",
    "sample_submission",
    "sell_prices",
]

# TARGET
TARGET = 'demand'

# VALIDATION
VALIDATION = 'HOLD_OUT'

# Only use group k-fold
# GROUPS = 
GROUPS = False

FOLD = 4
METRIC = "auc"
LOSS = "multi_logloss"
ID_NAME = "PassengerId"

# モデルを回す直前のtrainとtestをアウトプットする
OUTPUT_USE_DF = True
#OUTPUT_USE_DF = False

# SHAP値を計算する
#CALC_SHAP = True
CALC_SHAP = False

# TIME
import datetime
NOW = datetime.datetime.now()
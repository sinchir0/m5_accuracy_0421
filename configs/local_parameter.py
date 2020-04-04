# CASE
CASE = "case2"

# 動作確認のみ
#OPE_CHECK =True
OPE_CHECK =False

# TIME(元がGMTのため、日本時間に変換)
import datetime
NOW = datetime.datetime.now() + datetime.timedelta(hours=9)

# PATH
ROOT_PATH = "/home/jupyter/kaggle_pipeline_m5_accuracy"
DATA_PATH = f"{ROOT_PATH}/data/input"
FEATURES_PATH = f"{ROOT_PATH}/features"
MODEL_PASS = f"{ROOT_PATH}/model_output"
SUB_PASS = f"{ROOT_PATH}/submission"

MAKE_PATH = f"{MODEL_PASS}/{CASE}"
PATH_W = f"{MODEL_PASS}/{CASE}/{CASE}_{NOW:%Y%m%d%H%M%S}_lgb_score.txt"

# USE_ALL_DATA Including PrivateLBData 
#USE_ALL_DATA = True
USE_ALL_DATA = False

# Trick to avoid memory spike when LightGBM converts everything to float32:
# See https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53773
# 何故かやるとスコアが落ちる・・・
#BINARY_CHANGE = True
BINARY_CHANGE = False

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

# label_encoderでラベルエンコーディングしないで欲しいcolumnはここに追加
EXCLUDE_COLUMNS = [
    'id',
    'date'
]

# data_allを生成時にlabel encodeも実施
AUTO_LABEL_ENCODER_IN_LOAD_DATASETS_AND_TARGET = True
#AUTO_LABEL_ENCODER_IN_LOAD_DATASETS_AND_TARGET = False

# fit時にとにかくlabel encodeを行う
#AUTO_LABEL_ENCODER = True
AUTO_LABEL_ENCODER = False

# モデルを回す直前のtrainとtestをアウトプットする
#OUTPUT_USE_DF = True
OUTPUT_USE_DF = False

# InfをNaNに変換するかどうか
#INF_TO_NAN = True
INF_TO_NAN = False

# SHAP値を計算する
#CALC_SHAP = True
CALC_SHAP = False


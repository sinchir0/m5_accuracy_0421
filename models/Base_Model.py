#https://www.kaggle.com/sinchir0/dsb-ens-case14?scriptVersionId=27445829

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import logging
import shap

from configs.local_parameter import *
from logs.logger import *
from utils import *

class Base_Model(object):
    
    def __init__(self, train_df, test_df, categoricals=[], verbose=True, valid_df = None):
        self.train_df = train_df
        self.test_df = test_df
        self.features = FEATURE
        self.features.extend(USE_ORG_FEATURE)
        if VALIDATION != 'HOLD_OUT':
            self.n_splits = FOLD
        self.categoricals = categoricals
        self.target = TARGET
        self.X = self.train_df[self.features]
        self.y = self.train_df[self.target]
        if GROUPS:
            self.groups = np.array(self.train_df[groups].values)
        if VALIDATION != 'HOLD_OUT':
            self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        if VALIDATION != 'HOLD_OUT':
            self.y_pred, self.score, self.model, self.oof_pred, self.models = self.fit_cv()
        elif VALIDATION == 'HOLD_OUT':
            self.valid_df = valid_df
            self.y_pred, self.score, self.model = self.fit_hold()
        
    def train_model(self, train_set, val_set):
        raise NotImplementedError
    
    def get_cv(self): 
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        #cv = stratified_group_k_fold(X, y, groups, k=5)
        return cv.split(self.train_df, self.train_df[self.target])
        #return stratified_group_k_fold(self.X, self.y, self.groups, k=5)
    
    def get_params(self):
        raise NotImplementedError
        
    def feature_importance(self):
        raise NotImplementedError
        
    def feature_importance_hold(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
    
    def output_use_df(self,train_df,test_df,valid_df = None):
        train_df.to_pickle(f'{MODEL_PASS}/{CASE}/train_{CASE}_{NOW:%Y%m%d%H%M%S}.pkl'
        #    index=False
        )
        
        if valid_df is not None:
            valid_df.to_pickle(f'{MODEL_PASS}/{CASE}/valid_{CASE}_{NOW:%Y%m%d%H%M%S}.pkl'
                           # index=False
                           )

        test_df.to_pickle(f'{MODEL_PASS}/{CASE}/test_{CASE}_{NOW:%Y%m%d%H%M%S}.pkl'
        #    index=False
        )
    
    def calc_shap(self, x_train, model, shap_sampling = 10000):
        raise NotImplementedError
        
    def calc_score(self, y_pred, y):
        return np.sqrt(mean_squared_error(y_pred, y))
    
    def fit_cv(self):
        print(f"Start {VALIDATION} {FOLD}_fold")
        logging.debug(f"Start {VALIDATION} {FOLD}_fold")
        
        train_pred = np.zeros((len(self.train_df), ))
        oof_pred = np.zeros((len(self.train_df), ))
        y_pred = np.zeros((len(self.test_df), ))
        models = []
        
        if OUTPUT_USE_DF:
            self.output_use_df(self.train_df,self.test_df)
        
        logging.debug(f"params:{self.params}")
        
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            
            print('Fold:',fold)
            print(f"train_idx:{train_idx}")
            print(f"val_idx:{val_idx}")
            logging.debug(f"Fold:{fold}")

            #train,fit
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            
            model = self.train_model(train_set, val_set)
            
            #predict(to train,valid) 今の所意味なし
            conv_x_train = self.convert_x(x_train)
            conv_x_val = self.convert_x(x_val)
            
            train_pred[train_idx] = model.predict(conv_x_train).reshape(train_pred[train_idx].shape)
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            
            # predict(to test)
            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
            
            #log
            logging.debug(f'best_iteration:{model.best_iteration}')
            train_score = f'train fold {fold} : {roc_auc_score(y_train, train_pred[train_idx])}'
            make_log(MAKE_PATH, PATH_W, train_score)
            logging.debug(train_score)
            
            valid_score = f'valid fold {fold}: {roc_auc_score(y_val, oof_pred[val_idx])}'
            make_log(MAKE_PATH, PATH_W, valid_score)
            logging.debug(valid_score)
            
            models.append(model)
            
            #特徴量重要度の算出
            fold_importance = self.feature_importance(model,fold)
            #fold_importance.rename(columns={'importance': fold},inplace=True)
            if fold == 0:
                feature_importances = fold_importance
            else:
                feature_importances = pd.merge(feature_importances,fold_importance,on='feature')
                
            #SHAP値の計算 本当は画像をjpgで保存したい
            if CALC_SHAP:
                print("Start CALC_SHAP")
                explainer, shap_values = self.calc_shap(x_train, model, 10000)
                shap.summary_plot(shap_values[1], x_train)
                print("Finish")
 
        score = roc_auc_score(self.train_df[self.target], oof_pred)
    
        #log
        score_log = f'Oof auc score is: : {score}'
        make_log(MAKE_PATH, PATH_W, score_log)
        logging.debug(score_log)
        
        #特徴量重要度の計算
        feature_importances.set_index("feature",inplace=True)
        feature_importances["mean"] = feature_importances.mean(axis='columns')
        feature_importances.sort_values("mean",ascending=False,inplace=True)
        print(feature_importances)
        
        #特徴量重要度の保存
        feature_importances.to_csv(f'{MODEL_PASS}/{CASE}/FTI_{CASE}_{NOW:%Y%m%d%H%M%S}_{score}.csv')
        logging.debug(feature_importances)
        
        #oof_predの保存
        oof_pred_df = pd.DataFrame({
            CASE :  oof_pred
        })
        oof_pred_df.to_pickle(f"{MODEL_PASS}/{CASE}/oof_pred_{CASE}_{NOW:%Y%m%d%H%M%S}_{score}.pkl")
       
        return y_pred, score, model, oof_pred, models
    
    def fit_hold(self):
        print(f"Start {VALIDATION}")
        logging.debug(f"Start {VALIDATION}")
        
        if AUTO_LABEL_ENCODER:
            print("Start auto_label_encoder")
        
            self.train_df, self.valid_df, self.test_df = auto_label_encoder(
                self.train_df,
                self.valid_df, 
                self.test_df,
                exclude_columns
            )

        print("Finish")
        
        if OUTPUT_USE_DF:
            print("Start OUTPUT_USE_DF")
            self.output_use_df(
                train_df = self.train_df,
                test_df = self.test_df,
                valid_df = self.valid_df
            )
            print("Finish")
        
        train_pred = np.zeros((len(self.train_df), ))
        valid_pred = np.zeros((len(self.valid_df), ))
        #oof_pred = np.zeros((len(self.train_df), ))
        y_pred = np.zeros((len(self.test_df), ))
        models = []
        
        logging.debug(f"params:{self.params}")
        
        #train,fit
        x_train, x_val = self.train_df[self.features], self.valid_df[self.features]
        y_train, y_val = self.train_df[self.target], self.valid_df[self.target]
        
        if BINARY_CHANGE:
            to_float32 =  x_train.select_dtypes("float").columns.tolist() 
            x_train[to_float32] = x_train[to_float32].astype("float32")
            x_val[to_float32] = x_val[to_float32].astype("float32")
        
        train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
        
        print("Start fit")
        model = self.train_model(train_set, val_set)
        print("Finish")
        
        # predict(to train,valid) 今の所意味なし
        conv_x_train = self.convert_x(x_train)
        conv_x_val = self.convert_x(x_val)
        
        print("Start predict")
        train_pred = model.predict(conv_x_train).reshape(train_pred.shape)
        valid_pred = model.predict(conv_x_val).reshape(valid_pred.shape)
        print("Finish")
        
        # predict(to test)
        x_test = self.convert_x(self.test_df[self.features])
        y_pred = model.predict(x_test).reshape(y_pred.shape)
        
        #log
        logging.debug(f'best_iteration:{model.best_iteration}')
        train_score = f'train_{self.calc_score(y_train, train_pred)}'
        make_log(MAKE_PATH, PATH_W, train_score)
        logging.debug(train_score)

        valid_score = f'valid_{self.calc_score(y_val, valid_pred)}'
        make_log(MAKE_PATH, PATH_W, valid_score)
        logging.debug(valid_score)

        #models.append(model)

        #SHAP値の計算 本当は画像をjpgで保存したい
        if CALC_SHAP:
            print("Start CALC SHAP")
            explainer, shap_values = self.calc_shap(x_train, model, 10000)
            shap.summary_plot(shap_values[1], x_train)
            print("finish")
        
        #log
        score_log = f'RMSSE score train : {train_score}, {valid_score}'
        make_log(MAKE_PATH, PATH_W, score_log)
        print(score_log)
        logging.debug(score_log)
        
        #特徴量重要度の計算
        feature_importances = self.feature_importance_hold(model)
        #feature_importances.set_index("feature",inplace=True)
        #feature_importances["mean"] = feature_importances.mean(axis='columns')
        feature_importances.sort_values("importance",ascending=False,inplace=True)
        print(feature_importances)
        
        #特徴量重要度の保存
        feature_importances.to_csv(f'{MODEL_PASS}/{CASE}/FTI_{CASE}_{NOW:%Y%m%d%H%M%S}_{valid_score}.csv')
        logging.debug(feature_importances)
        
        #oof_predの保存
        #oof_pred_df = pd.DataFrame({
        #    CASE :  oof_pred
        #})
        #oof_pred_df.to_pickle(f"{MODEL_PASS}/{CASE}/oof_pred_{CASE}_{NOW:%Y%m%d%H%M%S}_{score}.pkl")
       
        return y_pred, valid_score, model
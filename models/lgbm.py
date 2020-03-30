from models.Base_Model import *
import lightgbm as lgb
from logs.logger import log_evaluation


class Lgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        # ロガーの作成
        logger = logging.getLogger('main')
        callbacks = [log_evaluation(logger, period=30)]
        
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, 
                         train_set, 
                         valid_sets=[train_set, val_set], 
                         verbose_eval=verbosity,
                         callbacks=callbacks
                        )
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        if self.categoricals != []:
            train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        else:
            train_set = lgb.Dataset(x_train, y_train)
            val_set = lgb.Dataset(x_val, y_val)
        return train_set, val_set
        
    def get_params(self):
#         params = {
#             "num_boost_round" : 5000,
#             'boosting_type': 'gbdt',
#             'objective': 'regression',
#             'metric': 'rmse',
#             'subsample': 0.75,
#             'subsample_freq': 1,
#             'learning_rate': 0.01,
#             'feature_fraction': 0.9,
#             'max_depth': -1,
#             'lambda_l1': 0.2,  
#             'lambda_l2': 1,
#             'early_stopping_rounds': 100,
#             #"min_data_in_leaf":16,
#             "verbose" : -1
#         }
        
        params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': SEED,
        'learning_rate': 0.1,
        'bagging_fraction': 0.75,
        'bagging_freq': 10, 
        'colsample_bytree': 0.75,
        'num_boost_round' : 2500, 
        'early_stopping_rounds' : 50
    }
        
        return params
    
    def feature_importance(self,model,fold):
        fold_importance = pd.DataFrame(list(zip(self.X.columns, model.feature_importance())),
                                           columns=['feature', fold])
        return fold_importance
    
    def feature_importance_hold(self,model):
        fold_importance = pd.DataFrame(list(zip(self.X.columns, model.feature_importance())),
                                           columns=['feature','importance'])
        return fold_importance
    
    def calc_shap(self, x_train, model, shap_sampling = 10000):
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train[:shap_sampling]) 
        
        return explainer, shap_values
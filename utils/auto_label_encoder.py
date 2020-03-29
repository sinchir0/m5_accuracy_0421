import pandas as pd
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
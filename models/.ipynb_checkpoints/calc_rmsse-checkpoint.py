import pandas as pd
import numpy as np
from configs.local_parameter import *
from scipy.sparse import csr_matrix
import gc
import pickle

print("start calc weight of wrmsse")

sales_train_val = pd.read_pickle(f'{DATA_PATH}/sales_train_validation.pkl')
submission = pd.read_pickle(f'{DATA_PATH}/sample_submission.pkl')
data = pd.read_pickle(f'{DATA_PATH}/data.pkl')

# 予測期間とitem数の定義 / number of items, and number of prediction period
NUM_ITEMS = sales_train_val.shape[0]  # 30490
DAYS_PRED = submission.shape[1] - 1  # 28

# sales_train_valからidの詳細部分(itemやdepartmentなどのid)を重複なく一意に取得しておく。(extract a detail of id columns)
product = sales_train_val[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

# create weight matrix
# levelに応じて、flagを立てたマトリクスを立てる
# 最後に転置をしているから、1~30490が列、各レベルごとの条件のflagが行にくる
# 例えば、level1の場合、　全て１が立っている行が一つだけある
# 一方、level2の場合、CA  TX  WIがそれぞれ１行ずつある。flagが立っているのは、該当する列。

weight_mat = np.c_[
    # level 1 Unit sales of all products, aggregated for all stores/states
    np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
    # level 2 Unit sales of all products, aggregated for each State               
    pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
    # level 3 Unit sales of all products, aggregated for each store               
    pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
    # level 4 Unit sales of all products, aggregated for each category               
    pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
    # level 5 Unit sales of all products, aggregated for each department                
    pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
    # level 6 Unit sales of all products, aggregated for each State and category               
    pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
    # level 7 Unit sales of all products, aggregated for each State and department               
    pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
    # level 8 Unit sales of all products, aggregated for each store and category               
    pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
    # level 9 Unit sales of all products, aggregated for each store and department               
    pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
    # level 10 Unit sales of product x, aggregated for all stores/states               
    pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
    # level 11 Unit sales of product x, aggregated for each State               
    pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values,
    # level 12 Unit sales of product x, aggregated for each store               
    np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
                   ].T

# 疎行列の圧縮を行なっている
# Compressed Sparse Rowの略。圧縮行格納方式。
weight_mat_csr = csr_matrix(weight_mat)
del weight_mat; gc.collect()

def weight_calc(data,product):
    
    # calculate the denominator of RMSSE, and calculate the weight base on sales amount
    
    sales_train_val = pd.read_pickle(f'{DATA_PATH}/sales_train_validation.pkl')

    d_name = ['d_' + str(i+1) for i in range(1913)]

    ### sales_train_val = weight_mat_csr * sales_train_val[d_name].values
    # weight_matのshapeは(42840, 30490)
    # sales_train_valのshapeは(30490, 1913)
    # つまり、weight_matの１行ごとに対して、sales_train_valの1列をかける行列演算を行なっている。
    # 1913列は与えられた期間である1941日からvalidationに使う28日を引いた数。
    # 42840行は、それぞれのLEVEL毎に計算が必要な行の総和
    # つまりこの掛け算によって、各日付（列方向）毎に、各レベルの商品が何個売れたかがわかる。
    
    # 出力例
    # [[32631 31749 23783 ... 40517 48962 49795]
    #  [14195 13805 10108 ... 17095 21834 23187]
    #  [ 9438  9630  6778 ... 10615 12266 12282]
    #  ...
    #  [    0     6     0 ...     0     1     0]
    #  [    0     0     0 ...     3     1     3]
    #  [    0     0     0 ...     0     0     0]]
    
    sales_train_val = weight_mat_csr * sales_train_val[d_name].values

    # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
    # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
    
    #####df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))
    
    ### np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1))
    # このコードで、１~1913の列を、行方向に42840個(今回の重み計算に使う全データ数)に拡張している
    
    ### ((sales_train_val>0) * np.tile(np.arange(1,1913),(weight_mat_csr.shape[0],1)))
    # sales_train_valの数値が０ではない部分は、そのまま1~1913の数字を残す
    # sales_train_valの数値が０の部分は、値を0にする
    # つまり売り上げがある日はそのまま、無い日を０にしている
    
    # df_tmpは、売り上げがある日はそのままの数字、無い日は０が入っている行列
    # 一行目が[1,2,3,0,5,6]とかなら、1913日のうち、4日目は売り上げがないことを表す。
    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))
    
    ##### start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
    
    ### np.where(df_tmp==0,9999,df_tmp)
    # df_tmpの0を9999に一度変換(0を考慮しないための計算)
    
    ### len(np.min(np.where(df_tmp==0,9999,df_tmp),axis=1))
    # 各列の最小値を取得
    # 売り上げが無い日は現在９９９９なので、この最小値は、売り上げが初めて０じゃなくなったタイミング
    
    ### start_no
    # 各列の最小値から１を引いた数字を取得
    # 評価に使われる42840個の行に関して、売り上げ開始日の1日前を表す
    
    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
    
    #####flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1
    
    ###np.diag(1/(start_no+1))
    # np.diagは対角行列を返す
    # もしstart_noが0(1日目から販売している)場合、この対角成分には１が入る
    # そうじゃない場合には１より小さい数字が入る
    
    ### np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1))
    # np.tileはnp.arange(1,1914)を（行方向の繰り返し数、列方向の繰り返し数） = (weight_mat_csr.shape[0],1)でarrayを生成している。
    # weight_mat_csr.shape[0] = 42840
    # つまりnp.arange(1,1914)を行方向に42840回繰り返したarrayを生成している
    
    ### np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1
    # np.diag(1/(start_no+1)) と　np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))の内積が１より小さかったらTrue
    # np.diag(1/(start_no+1))は対角行列、成分は1日目から売り上げがあったら１、そうじゃなかったら１未満
    # つまりこの行列(flag)は42480のデータに関して、初日から売り上げ数があるセルにはFalse,ないセルには
    # Trueが入る。行42480の評価対象　列1〜1913の売り上げ日数
    
    flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1

    sales_train_val = np.where(flag,np.nan,sales_train_val)

    # denominator of RMSSE / RMSSEの分母
    weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1913-start_no)

    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum)
    df_tmp = df_tmp[product.id].values
    
    weight2 = weight_mat_csr * df_tmp 

    weight2 = weight2/np.sum(weight2)

    del sales_train_val
    gc.collect()
    
    return weight1, weight2

#weight1, weight2 = weight_calc(data,product)
#np.save(f'{WEIGHT_PASS}/weight1', weight1)
#np.save(f'{WEIGHT_PASS}/weight2', weight2)

weight1 = np.load(f'{WEIGHT_PASS}/weight1.npy')
weight2 = np.load(f'{WEIGHT_PASS}/weight2.npy')

#def wrmsse(preds, data):
def wrmsse_for_feval(preds, data):
    
    # this function is calculate for last 28 days to consider the non-zero demand period
    
    # actual obserbed values / 正解ラベル
    y_true = data.get_label()
    
    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    # number of columns
    num_col = DAYS_PRED
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    
    train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / weight1) * weight2)
    
    return 'wrmsse', score, False

def wrmsse_for_score(y_true,preds,TRAIN=False):
    y_true = y_true.to_numpy()
    #preds = preds.to_numpy()

    # number of columns
    num_col = DAYS_PRED

    if TRAIN:
        print(DAYS_PRED)
        print(y_true.shape[0])
        #num_col = y_true.shape[0] / NUM_ITEMS
        num_col = 1913
        print(num_col)
    
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
          
    train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / weight1) * weight2)
    
    return score

def wrmsse_simple(preds, data):
    
    # actual obserbed values / 正解ラベル
    y_true = data.get_label()
    
    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    # number of columns
    num_col = DAYS_PRED
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
          
    train = np.c_[reshaped_preds, reshaped_true]
    
    weight2_2 = weight2[:NUM_ITEMS]
    weight2_2 = weight2_2/np.sum(weight2_2)
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) /  weight1[:NUM_ITEMS])*weight2_2)
    
    return 'wrmsse', score, False

print("end")
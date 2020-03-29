import sys
import os

args = sys.argv

#　ダブルクォーテーションを追加
args[2] = str('\"') + args[2] + str('\"')

os.system(f"kaggle competitions submit -c m5-forecasting-accuracy -f submission/{args[1]} -m {args[2]}")
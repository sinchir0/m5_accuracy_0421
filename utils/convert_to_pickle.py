import sys
sys.path.append('.')

import pandas as pd
import pickle
from configs.local_parameter import *

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for t in TO_PICKLE_TARGET:
    (pd.read_csv(f'{ROOT_PATH}/data/input/' + t + '.' + extension, encoding="utf-8"))\
        .to_pickle(f'{ROOT_PATH}/data/input/' + t + '.pkl')

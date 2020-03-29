import argparse
import inspect
import pickle
import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def get_arguments(description='default'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--force', '-f', action='store_true', help='Overwrite existing files'
    )
    return parser.parse_args()


def get_features(namespace):
    '''
    クラス型で定義され、更にsubclassで定義されたfeaturegはあって
    abstract class メソッドではないもののみ、yieldで返す。
    多分、ここでcreate内の特徴量生成のclassのみに絞っている。
    '''
    for k, v in ({k: v for k, v in namespace.items()}).items():
        if inspect.isclass(v) and issubclass(v, Feature) \
                and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    """
    get_featuresをかいくぐった変数のみここに入る。
    既にファイルが存在していたらskip.
    それ以外は、runして、saveする。
    runとsaveについてはclass Featureの中身を確認。
    """
    
    for f in get_features(namespace):
        print(f)
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        self.name = self.__class__.__name__
#         #もし変数名が大文字なら
#         if self.__class__.__name__.isupper():
#             #小文字に変換
#             self.name = self.__class__.__name__.lower()
#         else:
#             print(f"before:{self.__class__.__name__}")
#             self.name = re.sub(
#                 "([A-Z])",
#                 lambda x: "_" + x.group(1).lower(), self.__class__.__name__
#             ).lstrip('_')

        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        #self.train_path = Path(self.dir) / f'{self.name}_train.feather'
        #self.test_path = Path(self.dir) / f'{self.name}_test.feather'
        self.train_path = Path(self.dir) / f'{self.name}_train.pkl'
        self.test_path = Path(self.dir) / f'{self.name}_test.pkl'

    def run(self):
        with timer(self.name):
            #この関数をcreate.pyのそれぞれの定義に継承？して、create_featureは
            #特徴量ごとの動作にしている。
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train = self.train.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)
        #self.train.to_feather(str(self.train_path))
        #self.test.to_feather(str(self.test_path))
        self.train.to_pickle(str(self.train_path))
        self.test.to_pickle(str(self.test_path))


    def load(self):
        #self.train = pd.read_feather(str(self.train_path))
        #self.test = pd.read_feather(str(self.test_path))
        self.train = pd.read_pickle(str(self.train_path))
        self.test = pd.read_pickle(str(self.test_path))
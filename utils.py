import os
import torch
import random
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

TASK_DICT = {
    'blood': "Did the person donate blood? Yes or no?",
    'credit-g': "Does this person receive a credit? Yes or no?",
    'diabetes': "Does this patient have diabetes? Yes or no?",
    'heart': "Does the coronary angiography of this patient show a heart disease? Yes or no?",
    'adult': "Does this person earn more than 50000 dollars per year? Yes or no?",
    'bank': "Does this client subscribe to a term deposit? Yes or no?",
    'car': "How would you rate the decision to buy this car? Unacceptable, acceptable, good or very good?",
    'communities': "How high will the rate of violent crimes per 100K population be in this area. Low, medium, or high?",
    'myocardial': "Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no?",
    "Amazon": "Does the resource was approved? Yes or no?",
}


class Feature_type_recognition():
    def __init__(self):
        self.df = None

    def detect_TIMESTAMP(self, col):
        try:
            ts_min = int(float(self.df.loc[~(self.df[col] == '') & (self.df[col].notnull()), col].min()))
            ts_max = int(float(self.df.loc[~(self.df[col] == '') & (self.df[col].notnull()), col].max()))
            datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime('%Y-%m-%d %H:%M:%S')
            datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime('%Y-%m-%d %H:%M:%S')
            if datetime_min > '2000-01-01 00:00:01' and datetime_max < '2030-01-01 00:00:01' and datetime_max > datetime_min:
                return True
        except:
            return False

    def detect_DATETIME(self, col):
        is_DATETIME = False
        if self.df[col].dtypes == object or str(self.df[col].dtypes) == 'category':
            is_DATETIME = True
            try:
                pd.to_datetime(self.df[col])
            except:
                is_DATETIME = False
        return is_DATETIME

    def get_data_type(self, col):
        if self.detect_DATETIME(col):
            return 'cat'
        if self.detect_TIMESTAMP(col):
            return 'cat'
        if self.df[col].dtypes == object or self.df[col].dtypes == bool or str(self.df[col].dtypes) == 'category':
            return 'cat'
        if 'int' in str(self.df[col].dtype) or 'float' in str(self.df[col].dtype):
            if self.df[col].nunique() < 15:
                return 'cat'
            return 'num'

    def fit(self, df):
        self.df = df
        self.num = []
        self.cat = []
        self.bin = []
        for col in self.df.columns:
            cur_type = self.get_data_type(col)
            if (cur_type == 'num'):
                self.num.append(col)
            elif (cur_type == 'cat'):
                self.cat.append(col)
            elif (cur_type == 'bin'):
                self.bin.append(col)
            else:
                raise RuntimeError('error feature type!')
        return self.cat, self.bin, self.num


def load_single_data_all(table_file, target=None, auto_feature_type=None, encode_cat=False):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)

        if not target:
            target = df.columns.tolist()[-1]
        if not auto_feature_type:
            auto_feature_type = Feature_type_recognition()

        # Delete the sample whose label count is 1 or label is nan
        count_num = list(df[target].value_counts())
        count_value = list(df[target].value_counts().index)
        delete_index = []
        for i, cnt in enumerate(count_num):
            if cnt <= 1:
                index = df.loc[df[target] == count_value[i]].index.to_list()
                delete_index.extend(index)
        df.drop(delete_index, axis=0, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, subset=[target], inplace=True)

        y = df[target]
        X = df.drop([target], axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)

        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y, index=X.index, name=target)
    else:
        raise RuntimeError('no such data file!')

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower())

    if len(bin_cols) > 0:
        for col in bin_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[bin_cols] = X[bin_cols].astype(str).applymap(
            lambda x: 1 if x.lower() in ['yes', 'true', '1', 't'] else 0).values
        for col in bin_cols:
            if X[col].nunique() <= 1:
                raise RuntimeError('bin feature process error!')

    X = X[bin_cols + num_cols + cat_cols]

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X),
                                                                                                     len(attribute_names),
                                                                                                     len(cat_cols),
                                                                                                     len(bin_cols),
                                                                                                     len(num_cols), (
                                                                                                                 y == 1).sum() / len(
            y)))
    return X, y, cat_cols, num_cols, bin_cols, target


def get_dataset(data_name, shot, seed):
    file_name = f"./data/{data_name}.csv"

    df = pd.read_csv(file_name)

    default_target_attribute = df.columns[-1]

    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in
                             df.dtypes.tolist()][:-1]

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()

    label_list = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    assert (shot <= 128)  # We only consider the low-shot regimes here
    X_new_train = X_train.copy()
    X_new_train[default_target_attribute] = y_train
    sampled_list = []

    remainder = shot % len(np.unique(y_train))
    for _, grouped in X_new_train.groupby(default_target_attribute):
        sample_num = shot // len(np.unique(y_train))
        if remainder > 0:
            sample_num += 1
            remainder -= 1
        grouped = grouped.sample(sample_num, random_state=seed)
        sampled_list.append(grouped)
    X_balanced = pd.concat(sampled_list)
    X_train_ = X_balanced.drop([default_target_attribute], axis=1)
    y_train_ = X_balanced[default_target_attribute].to_numpy()

    return df, X_train, X_train_, X_test, y_train, y_train_, y_test, default_target_attribute, label_list, categorical_indicator


def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(pred_probs, answers, multiclass=False):

    if multiclass == False:
        result_auc = roc_auc_score(answers, pred_probs[:, 1])
    else:
        result_auc = roc_auc_score(answers, pred_probs, multi_class='ovr', average='macro')
    return result_auc

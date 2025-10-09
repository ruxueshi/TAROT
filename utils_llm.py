import torch
import openai
import datetime
import pandas as pd


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler


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
            # if self.df[col].nunique() == 2:
            #     return 'bin'
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


def load_single_data_all(path, target=None, auto_feature_type=None):
    # df = table
    df = pd.read_csv(path)

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

    X = df.drop([target], axis=1)
    all_cols = [col.lower() for col in X.columns.tolist()]

    X.columns = all_cols
    # divide cat/bin/num feature
    cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)

    return cat_cols, num_cols, bin_cols


def data_process(X, y, num_cols, cat_cols, bin_cols, encode_cat=False):
    # encode target label
    y = LabelEncoder().fit_transform(y)
    y = pd.Series(y, index=X.index, name="label")

    all_cols = [col.lower() for col in X.columns.tolist()]

    X.columns = all_cols
    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)

        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])

        else:
            # X[cat_cols] = X[cat_cols].astype(str).str.lower()
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

    return X, y


def query_gpt(prompt, keys, max_tokens=30, temperature=0, max_try_num=10, model="gpt-4o-mini"):
    openai.api_key = keys

    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    result = response.choices[0].message.content

    return result


def compute_loss(loss_fn, logits, y, num_class):
    if y is not None:
        # compute classification loss
        if num_class == 2:
            if isinstance(y, pd.Series):
                y_ts = torch.tensor(y.values).cuda().float()
            else:
                y_ts = y.float().cuda()
            loss = loss_fn(logits.flatten(), y_ts)
        else:
            if isinstance(y, pd.Series):
                y_ts = torch.tensor(y.values).cuda().long()
            else:
                y_ts = y.long().cuda()
            loss = loss_fn(logits, y_ts)

        loss = loss.mean()
    else:
        loss = None

    return loss


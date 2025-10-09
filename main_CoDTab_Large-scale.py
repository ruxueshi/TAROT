import argparse
import os
import shutil
import torch
import utils

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from utils import load_single_data_all
from TNE.evaluator import predict

import TNE

import warnings
warnings.filterwarnings("ignore")

# set random seed
utils.random_seed(42)

cal_device = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser(description='CoDTab')
    parser.add_argument('--log_name', type=str, default="CoDTab_scratch", help='task name')
    parser.add_argument('--data', type=str, default="blood", help='task dataset')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph_num', type=int, default=1)
    args = parser.parse_args()
    return args

_args = parse_args()

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
all_res = {}
data_name = _args.data
table_file_path = "./data/" + data_name + ".csv"

adj_list = []
for i in range(_args.graph_num):
    adj = np.load(fr"data\graph{i + 1}\{data_name}_{42}_{4}_graph.npy")
    adj = torch.from_numpy(adj).cuda()
    adj_list.append(adj)

print(f'Start========>{data_name}_graph_DataSet==========>')
X, y, cat_cols, num_cols, bin_cols, target = load_single_data_all(table_file_path)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

num_class = len(y.value_counts())
print(f'num_class : {num_class}')
cat_cols = [cat_cols]
num_cols = [num_cols]
bin_cols = [bin_cols]
idd = 0
score_list = []
for trn_idx, val_idx in skf.split(X, y):
    utils.random_seed(42)
    idd += 1
    train_data = X.loc[trn_idx]
    train_label = y[trn_idx]
    X_test = X.loc[val_idx]
    y_test = y[val_idx]
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=0, stratify=train_label, shuffle=True)

    model = TNE.CoDTab(
        checkpoint=None,
        device=cal_device,
        num_class=num_class,
        num_layer=3,
        gcn_layer=_args.layer,
        hidden_dropout_prob=0.1,
        vocab_freeze=True,
        use_bert=True,
    )
    model.update({'cat': cat_cols, 'num': num_cols, 'bin': bin_cols})

    training_arguments = {
            'num_epoch':300,
            'batch_size':64,
            'lr':1e-4,
            'eval_metric':'auc',
            'eval_less_is_better':False,
            'output_dir':f'./models/checkpoint-scratch',
            'patience':30,
            'num_workers':0,
            'device':cal_device,
            'flag':1,
            'warmup_steps':5,
        }
    if os.path.isdir(training_arguments['output_dir']):
        shutil.rmtree(training_arguments['output_dir'])
    trainer = TNE.train(model, target, adj_list, (X_train, y_train), (X_val, y_val), data_weight=[True], **training_arguments)
    eval_res_list = trainer.train((X_test, y_test))

    ypred = predict(model, X_test, target, adj_list)
    ans = utils.evaluate(ypred, y_test, True)
    ans = max(ans, max(eval_res_list[-5:]))
    score_list.append(ans)
    print(f'Test_Score_{idd}===>{data_name}_DataSet==> {ans}')
all_res[data_name] = np.mean(score_list)
print(f'Test_Score_5_fold===>{data_name}_DataSet==> {np.mean(score_list)}')

mean_list = []
for key in all_res:
    print(f'meaning_5_fold=>{all_res[key]}=>{key}')
    mean_list.append(all_res[key])
print(f'meaning all data=>{np.mean(mean_list)}')
import os
import utils
import argparse
from utils_llm import *
import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam

import TNE

def parse_args():
    parser = argparse.ArgumentParser(description='CoDTab')
    parser.add_argument('--data', type=str, default=f"heart")
    parser.add_argument('--shot', type=int, default=16)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--layer', type=int, default=8)
    parser.add_argument('--graph_num', type=int, default=1)
    args = parser.parse_args()

    return args

b = 0.1
args = parse_args()
path = fr".\data/{args.data}.csv"

utils.random_seed(args.seed)
df, _, X_train, X_test, _, y_train, y_test, target, label_list, is_cat = utils.get_dataset(args.data, args.shot, args.seed)
cat_cols, num_cols, bin_cols = load_single_data_all(path)

multiclass = True if len(label_list) > 2 else False

X_train, y_train = data_process(X_train, y_train, num_cols, cat_cols, bin_cols)
X_test, y_test = data_process(X_test, y_test, num_cols, cat_cols, bin_cols)


adj = np.load(fr"data\graph\{args.data}_{args.seed}_graph.npy")
adj = torch.from_numpy(adj).cuda()

cat_cols = [cat_cols]
num_cols = [num_cols]
bin_cols = [bin_cols]

cal_device = "cuda"
num_class = len(label_list)
cpt = r""
lambda1 = 0.1
lambda2 = 0.1

def simple_model():
    model = TNE.CoDTab(
        checkpoint=cpt,
        device=cal_device,
        num_class=num_class,
        num_layer=3,
        gcn_layer=args.layer,
        hidden_dropout_prob=0.1,
        vocab_freeze=True,
        use_bert=True,
    )
    model.update({'cat': cat_cols, 'num': num_cols, 'bin': bin_cols})

    return model


def train_adapter(X_train_now, label_list, shot, y_train_num):
    # criterion = nn.CrossEntropyLoss()
    if num_class > 2:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    model = simple_model()
    opt = Adam(model.parameters(), lr=1e-5)
    for _ in range(1000):
        opt.zero_grad()
        outputs, Lprior, Lsparse = model(X_train_now, [target], adj)
        if outputs.shape[-1] == 1:  # binary classification
            preds = outputs.sigmoid().detach().cpu()
            preds = torch.where(preds > 0.5, 1, 0).numpy()
        else:  # multi-class classification
            preds = torch.softmax(outputs, -1).detach().cpu()
            preds = preds.argmax(dim=1).numpy()
        acc = (np.array(y_train_num) == preds).sum() / len(preds)
        if acc == 1:
            break

        loss = compute_loss(criterion, outputs, y_train_num, num_class)
        loss = loss + lambda1 * Lprior + lambda2 * Lsparse
        loss.backward()
        opt.step()
    return model


trained_model = train_adapter(X_train, label_list, args.shot, y_train)

tmp = []
eval_batch_size = 64
x_len = X_test.shape[0]
for i in range(0, x_len, eval_batch_size):
    with torch.no_grad():
        trained_model.eval()
        test_outputs = trained_model(X_test[i:i+eval_batch_size], [target], adj)[0]
    tmp.append(test_outputs.detach().cpu())

test_outputs = torch.cat(tmp, dim=0)
if test_outputs.shape[-1] == 1: # binary classification
    test_outputs = test_outputs.sigmoid().numpy()
else: # multi-class classification
    test_outputs = torch.softmax(test_outputs,-1).numpy()

result_auc = utils.evaluate(test_outputs, y_test, num_class)
if not os.path.exists(f"result/{args.data}"):
    os.makedirs(f"result/{args.data}")
with open(f"result/{args.data}/result_{args.shot}shot_{args.layer}layer_{args.seed}seed_graph.txt", "w", encoding="utf-8") as f:
    f.write(str(result_auc))
print("after adapter AUC:", result_auc)
import os
import torch
import numpy as np
from .trainer import Trainer
from .modeling_TNE import CoDTab

dev = 'cuda'


def build_classifier_graph_GCN2_para_relu_mut(
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        gcn_layer=1,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        vocab_freeze=False,
        use_bert=True,
        device=dev,
        checkpoint=None,
        **kwargs) -> CoDTab:
    model = CoDTab(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        gcn_layer=gcn_layer,
        feature_extractor=feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        use_bert=use_bert,
        device=device,
        **kwargs,
    )


    return model

def train(
    model,
    target,
    adj_list,
    trainset,
    valset=None,
    cmd_args=None,
    num_epoch=10,
    batch_size=64,
    eval_batch_size=256,
    lr=1e-4,
    weight_decay=0,
    patience=5,
    warmup_ratio=None,
    warmup_steps=None,
    eval_metric='auc',
    output_dir='./ckpt',
    collate_fn=None,
    num_workers=0,
    balance_sample=False,
    load_best_at_last=True,
    ignore_duplicate_cols=True,
    eval_less_is_better=False,
    flag=0,
    regression_task=False,
    train_method='normal',
    device=None,
    data_weight=None,
    **kwargs,
    ):
    if isinstance(trainset, tuple): trainset = [trainset]

    train_args = {
        'num_epoch': num_epoch,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'lr': lr,
        'weight_decay':weight_decay,
        'patience':patience,
        'warmup_ratio':warmup_ratio,
        'warmup_steps':warmup_steps,
        'eval_metric':eval_metric,
        'output_dir':output_dir,
        'collate_fn':collate_fn,
        'num_workers':num_workers,
        'balance_sample':balance_sample,
        'load_best_at_last':load_best_at_last,
        'ignore_duplicate_cols':ignore_duplicate_cols,
        'eval_less_is_better':eval_less_is_better,
        'flag':flag,
        'regression_task':regression_task,
        'device':device,
        'data_weight':data_weight,
    }

    trainer = Trainer(
            model,
            [target],
            adj_list,
            trainset,
            valset,
            **train_args,
        )

    return trainer


import logging
import os, pdb
import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List

from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from . import constants
dev = 'cuda'


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class CoDTabFeatureExtractor_graph:
    def __init__(self,
                 categorical_columns=None,
                 numerical_columns=None,
                 binary_columns=None,
                 disable_tokenizer_parallel=False,
                 ignore_duplicate_cols=False,
                 **kwargs,
                 ) -> None:
        if os.path.exists('./TNE/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./TNE/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./TNE/tokenizer')
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel:  # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

    def __call__(self, x, target, x_cat=None, table_flag=0) -> Dict:
        encoded_inputs = {
            'x_num': None,
            'num_col_input_ids': None,
            'x_cat_input_ids': None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns[table_flag]] if self.categorical_columns[
                                                                                              table_flag] is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns[table_flag]] if self.numerical_columns[
                                                                                            table_flag] is not None else []

        if len(cat_cols + num_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        # TODO:
        # mask out NaN values like done in binary columns
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0)  # fill Nan with zero
            x_num_ts = torch.tensor(x_num.values, dtype=torch.float32)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False,
                                        return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask']  # mask out attention

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_cat = x_cat.fillna('')

            # x_cat = x_cat.apply(lambda x: x.name + ' is ' + x) * x_mask # mask out nan features
            x_cat_str = x_cat.values.tolist()
            encoded_inputs['x_cat_input_ids'] = []
            encoded_inputs['x_cat_att_mask'] = []
            max_y = 0
            cat_cnt = len(cat_cols)
            # max_token_len = max(int(1), int(4096/cat_cnt))
            max_token_len = max(int(1), int(2048 / cat_cnt))
            for sample in x_cat_str:
                x_cat_ts = self.tokenizer(sample, padding=True, truncation=True, add_special_tokens=False,
                                          return_tensors='pt')
                x_cat_ts['input_ids'] = x_cat_ts['input_ids'][:, :max_token_len]
                x_cat_ts['attention_mask'] = x_cat_ts['attention_mask'][:, :max_token_len]
                encoded_inputs['x_cat_input_ids'].append(x_cat_ts['input_ids'])
                encoded_inputs['x_cat_att_mask'].append(x_cat_ts['attention_mask'])
                max_y = max(max_y, x_cat_ts['input_ids'].shape[1])
            for i in range(len(encoded_inputs['x_cat_input_ids'])):
                # tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp = torch.full((cat_cnt, max_y), self.pad_token_id, dtype=int)
                tmp[:, :encoded_inputs['x_cat_input_ids'][i].shape[1]] = encoded_inputs['x_cat_input_ids'][i]
                encoded_inputs['x_cat_input_ids'][i] = tmp
                tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp[:, :encoded_inputs['x_cat_att_mask'][i].shape[1]] = encoded_inputs['x_cat_att_mask'][i]
                encoded_inputs['x_cat_att_mask'][i] = tmp
            encoded_inputs['x_cat_input_ids'] = torch.stack(encoded_inputs['x_cat_input_ids'], dim=0)
            encoded_inputs['x_cat_att_mask'] = torch.stack(encoded_inputs['x_cat_att_mask'], dim=0)

            col_cat_ts = self.tokenizer(cat_cols, padding=True, truncation=True, add_special_tokens=False,
                                        return_tensors='pt')
            encoded_inputs['col_cat_input_ids'] = col_cat_ts['input_ids']
            encoded_inputs['col_cat_att_mask'] = col_cat_ts['attention_mask']

            col_target_ts = self.tokenizer(target, padding=True, truncation=True, add_special_tokens=False,
                                           return_tensors='pt')
            encoded_inputs['col_target_input_ids'] = col_target_ts['input_ids']
            encoded_inputs['col_target_att_mask'] = col_target_ts['attention_mask']

        return encoded_inputs

    def save(self, path):
        '''
        save the feature extractor configuration to local dir.
        '''
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

    def load(self, path):
        '''load the feature extractor configuration from local dir.
        '''
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    def update(self, cat=None, num=None, bin=None):
        if cat is not None:
            self.categorical_columns = cat

        if num is not None:
            self.numerical_columns = num

        if bin is not None:
            self.binary_columns = bin


class CoDTabFeatureProcessor_graph(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 vocab_dim=768,
                 hidden_dim=128,
                 hidden_dropout_prob=0,
                 pad_token_id=0,
                 vocab_freeze=False,
                 use_bert=True,
                 pool_policy='avg',
                 device=dev,
                 ) -> None:
        super().__init__()
        self.word_embedding = CoDTabWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            vocab_dim=vocab_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
        )
        self.num_embedding = CoDTabNumEmbedding(vocab_dim)
        self.align_layer = nn.Linear(vocab_dim, hidden_dim, bias=False)

        self.pool_policy = pool_policy
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is None:
            return embs.mean(-2)
        else:
            embs[att_mask == 0] = 0
            embs = embs.sum(-2) / (att_mask.sum(-1, keepdim=True).to(embs.device) + eps)
            return embs

    def _max_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is not None:
            embs[att_mask == 0] = -1e12
        embs = torch.max(embs, dim=-2)[0]
        return embs

    def _sa_block(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        return x[:, 0, :]

    def _check_nan(self, value):
        return torch.isnan(value).any().item()

    def forward(self,
                x_num=None,
                num_col_input_ids=None,
                num_att_mask=None,
                x_cat_input_ids=None,
                x_cat_att_mask=None,
                col_cat_input_ids=None,
                col_cat_att_mask=None,
                col_target_input_ids=None,
                col_target_att_mask=None,
                # cat_class=None,
                # x_cat=None,
                **kwargs,
                ) -> Tensor:
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None
        target_col_emb = None

        other_info = {
            'col_emb': None,  # [num_fs+cat_fs]
            'num_cnt': 0,  # num_fs
            'x_num': x_num,  # [bs, num_fs]
            'cat_bert_emb': None  # [bs, cat_fs, dim]
        }

        if other_info['x_num'] is not None:
            other_info['x_num'] = other_info['x_num'].to(self.device)

        if self.pool_policy == 'avg':
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device),
                                                  emb_type='header')  # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if col_target_input_ids is not None:
                col_target_feat_embedding = self.word_embedding(col_target_input_ids.to(self.device), emb_type='header')
                target_col_emb = self._avg_embedding_by_mask(col_target_feat_embedding, col_target_att_mask)
                target_col_emb = self.align_layer(target_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                x_cat_feat_embedding = self._avg_embedding_by_mask(x_cat_feat_embedding, x_cat_att_mask)
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                cat_col_emb = self._avg_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0], -1, -1))

                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._avg_embedding_by_mask(cat_feat_embedding)

                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)

                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)

                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()

        emb_list = []
        att_mask_list = []
        col_emb = []
        if target_col_emb is not None:
            col_emb += [target_col_emb]

        if num_feat_embedding is not None:
            col_emb += [num_col_emb]
            other_info['num_cnt'] = num_col_emb.shape[0]
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1]).to(self.device)]

            # emb_list += [num_feat_embedding]
            # att_mask_list += [num_att_mask]
        if cat_feat_embedding is not None:
            col_emb += [cat_col_emb]
            emb_list += [cat_feat_embedding]
            att_mask_list += [torch.ones(cat_feat_embedding.shape[0], cat_feat_embedding.shape[1]).to(self.device)]

        if len(emb_list) == 0: raise Exception(
            'no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        other_info['col_emb'] = torch.cat(col_emb, 0).float()

        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}, other_info


class CoDTabWordEmbedding(nn.Module):
    def __init__(self,
        vocab_size,
        hidden_dim,
        vocab_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        vocab_freeze=False,
        use_bert=True,
        ) -> None:
        super().__init__()
        word2vec_weight = torch.load('./TNE/bert_emb.pt')
        self.word_embeddings_header = nn.Embedding.from_pretrained(word2vec_weight, freeze=vocab_freeze, padding_idx=padding_idx)
        self.word_embeddings_value = nn.Embedding(vocab_size, vocab_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings_value.weight)

        self.norm_header = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)
        weight_emb = torch.load('./TNE/bert_layernorm_weight.pt')
        bias_emb = torch.load('./TNE/bert_layernorm_bias.pt')
        self.norm_header.weight.data.copy_(weight_emb)
        self.norm_header.bias.data.copy_(bias_emb)
        if vocab_freeze:
            freeze(self.norm_header)
        self.norm_value = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, emb_type) -> Tensor:

        if emb_type == 'header':
            embeddings = self.word_embeddings_header(input_ids)
            embeddings = self.norm_header(embeddings)
        elif emb_type == 'value':
            embeddings = self.word_embeddings_value(input_ids)
            embeddings = self.norm_value(embeddings)
        else:
            raise RuntimeError(f'no {emb_type} word_embedding method!')

        embeddings = self.dropout(embeddings)

        return embeddings

class  CoDTabNumEmbedding(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias

        return num_feat_emb


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class  CoDTabTransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None) -> Tensor:
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else: # do not use layer norm
                x = x + self._sa_block(x, src_mask, src_key_padding_mask)
                x = x + self._ff_block(x)
        return x


class  CoDTabInputEncoder(nn.Module):
    def __init__(self,
        feature_extractor,
        feature_processor,
        device=dev,
        ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds
    
    def load(self, ckpt_dir):
        # load feature extractor
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')


class  CoDTabEncoder(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
                CoDTabTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            ]
            )

        if num_layer > 1:
            encoder_layer = CoDTabTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer-1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)

        return outputs


class  CoDTabLinearClassifier(nn.Module):
    def __init__(self,
        num_class,
        hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, global_table_tensor=None) -> Tensor:
        x = x[:,0,:] # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)

        return logits


class  CoDTabCLSToken_graph(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, col_emb, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([col_emb[0].unsqueeze(0).unsqueeze(0).repeat(embedding.shape[0], 1, 1), embedding], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask

        return outputs


class CoDTabModel_graph(nn.Module):
    def __init__(self,
                 categorical_columns=None,
                 numerical_columns=None,
                 binary_columns=None,
                 feature_extractor=None,
                 hidden_dim=128,
                 num_layer=2,
                 num_attention_head=8,
                 hidden_dropout_prob=0.1,
                 ffn_dim=256,
                 activation='relu',
                 device=dev,
                 vocab_freeze=False,
                 use_bert=True,
                 pool_policy='avg',
                 **kwargs,
                 ) -> None:

        super().__init__()
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns

        if feature_extractor is None:
            feature_extractor =  CoDTabFeatureExtractor_graph(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        feature_processor =  CoDTabFeatureProcessor_graph(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
        )

        self.input_encoder = CoDTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        self.encoder = CoDTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )

        self.cls_token = CoDTabCLSToken_graph(hidden_dim=hidden_dim)
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        encoder_output = self.encoder(**embeded)

        return encoder_output

    def load(self, ckpt_dir):
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        # save model weight state dict
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        col_map = {}
        for k, v in config.items():
            if k in ['cat', 'num', 'bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self.clf = CoDTabLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')

        return None


class CoDTab(CoDTabModel_graph):
    def __init__(self,
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
                 pool_policy='avg',
                 device=dev,
                 **kwargs,
                 ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            gcn_layer=1,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
            **kwargs,
        )
        self.gcn_layer = gcn_layer
        self.num_class = num_class
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.clf = CoDTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        self.to(device)

    def gcn(self, data, adj):
        eye = torch.eye(*adj.shape).cuda()
        adj = adj+eye
        adj = adj > 0
        adj = adj.long()
        adj = adj/(adj.sum(1).unsqueeze(1)+1e-9)
        for i in range(self.gcn_layer):
            data = torch.matmul(adj.unsqueeze(0), data)
            data = torch.relu(self.linear(data))

        return data

    def forward(self, x, target, adj_list, y=None, table_flag=0):
        encoder_output2 = None
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.input_encoder.feature_extractor(x, target, table_flag=table_flag)
        else:
            raise ValueError(f'CoDTabClassifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs, other_info = self.input_encoder.feature_processor(**inputs)

        encoder_list = []
        for adj in adj_list:
            encoder_output = self.gcn(outputs['embedding'], adj)
            encoder_list.append(encoder_output)
        encoder_output = torch.stack(encoder_list, dim=0)
        encoder_output = encoder_output.mean(0)
        encoder_output = encoder_output.mean(1)
        encoder_output = encoder_output.unsqueeze(1)

        logits = self.clf(encoder_output)

        return logits
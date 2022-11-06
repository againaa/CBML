#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import datetime
import math
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import build_graph, Data, split_validation
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict


class PositionEmbedding(nn.Module):
    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings  # 每一row資料有多長
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )


class Residual(Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 20
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.drop = True

    def forward(self, x):
        residual = x  # keep original input
        x = F.relu(self.d1(x))
        if self.drop:
            x = self.d2(self.dp(x))
        else:
            x = self.d2(x)
        out = residual + x
        return out


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """

        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            # scores = scores.masked_fill(mask, min_value)
            # print("score.shape=",(torch.softmax(scores, dim=0)).shape)
            # print("mask.shape=",mask.shape)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)

        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.cluster_num = opt.cluster_num
        self.memory_dim = opt.memory_dim
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding1 = nn.Embedding(1298, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_one1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_transform1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_concat = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.predict = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)

        self.rn = Residual()
        self.rn1 = Residual()
        # self.dropout = nn.Dropout(p=0.2)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1).cuda()
        self.multihead_attn1 = nn.MultiheadAttention(self.hidden_size, 1).cuda()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.optimizer2 = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.mem_M = torch.nn.Parameter(
            torch.rand((self.cluster_num, self.memory_dim), dtype=torch.float32, requires_grad=True).cuda())
        self.mem_Wa = torch.nn.Parameter(
            torch.rand((self.hidden_size * 2, self.memory_dim), dtype=torch.float32, requires_grad=True).cuda())
        self.mem_fc = torch.nn.Parameter(
            torch.rand((self.memory_dim, self.memory_dim), dtype=torch.float32, requires_grad=True).cuda())
        self.weight = torch.nn.Parameter(F.sigmoid(torch.rand((1), dtype=torch.float32, requires_grad=True)).cuda())


        self.local_update_target_weight_name = ['embedding.weight', 'embedding1.weight', 'rn.d1.weight', 'rn.d1.bias',
                                                'rn.d2.weight', 'rn.d2.bias', 'rn1.d1.weight', 'rn1.d1.bias',
                                                'rn1.d2.weight', 'rn1.d2.bias', 'multihead_attn.in_proj_weight',
                                                'multihead_attn.in_proj_bias', 'multihead_attn.out_proj.weight',
                                                'multihead_attn.out_proj.bias',
                                                'multihead_attn1.in_proj_weight', 'multihead_attn1.in_proj_bias',
                                                'multihead_attn1.out_proj.weight', 'multihead_attn1.out_proj.bias',
                                                'predict.weight', 'predict.bias']
        self.mem_name = ['mem_M', 'mem_Wa', 'mem_fc', 'linear_concat.weight', 'linear_concat.bias']

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        par = self.state_dict()
        for weight in self.parameters():
            # if kys[i] not in names:
            weight.data.uniform_(-stdv, stdv)
            # i+=1

    def attention(self, inp):
        # query = torch.tanh(torch.matmul(inp, self.mem_Wa))
        query = torch.matmul(inp, self.mem_Wa)
        score = torch.matmul(query, (self.mem_M).permute((1, 0)))
        return F.softmax(score, dim=-1)



    def compute_scores1(self, item_seq_hidden, cat_seq_hidden, mask, self_att=True, residual=True, k_blocks=2):
        ht_item = item_seq_hidden[
            torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        ht_cat = cat_seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        attn_output = item_seq_hidden
        for k in range(k_blocks):
            attn_output, attn_output_weights = self.multihead_attn(attn_output, attn_output, attn_output)
            if residual:
                attn_output = self.rn(attn_output)
        a_hn_item = attn_output[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        a_item = 0.52 * a_hn_item + (1 - 0.52) * ht_item
        attn_output1 = cat_seq_hidden
        for k in range(k_blocks):
            attn_output1, attn_output_weights1 = self.multihead_attn1(attn_output1, attn_output1, attn_output1)
            if residual:
                attn_output1 = self.rn1(attn_output1)
        a_hn_cat = attn_output1[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        a_cat = 0.52 * a_hn_cat + (1 - 0.52) * ht_cat

        a = torch.cat((a_item, a_cat), dim=-1)

        attention_vals = self.attention(a)
        attentive_cluster_reps = torch.tanh(torch.matmul(torch.matmul(attention_vals, self.mem_M), self.mem_fc))

        oi = self.linear_concat(torch.cat((a, attentive_cluster_reps), dim=-1))
        final_outputs = F.sigmoid(oi) * self.predict(torch.cat((a, attentive_cluster_reps), dim=-1))
        b = self.embedding.weight[1:]

        scores = torch.matmul(final_outputs, b.transpose(1, 0))
        return scores

    def forward(self, inputs, category, A):
        item_hidden = self.embedding(inputs)
        cate_hidden = self.embedding1(category)
        return item_hidden, cate_hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, dict):
    alias_inputs, A, items, mask, targets = data.get_slice(i)

    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    categoty = []
    for k in range(len(items)):
        cat = []
        for j in range(len(items[k])):
            itemid = (items[k][j]).tolist()
            if itemid in dict.keys():
                cat.append(dict[itemid])
            else:
                if itemid != 0:
                    print("itemid=", itemid)
                cat.append(0)
        categoty.append(cat)

    categotyi = trans_to_cuda(torch.Tensor(categoty).long())

    item_hidden, cate_hidden = model(items, categotyi, A)
    get_item = lambda i: item_hidden[i][alias_inputs[i]]
    item_seq_hidden = torch.stack([get_item(i) for i in torch.arange(len(alias_inputs)).long()])

    get_cate = lambda i: cate_hidden[i][alias_inputs[i]]
    cate_seq_hidden = torch.stack([get_cate(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores1(item_seq_hidden, cate_seq_hidden, mask)






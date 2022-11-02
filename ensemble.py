import numpy as np
import pandas as pd

import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

# from pytorch_lightning.metrics import Accuracy
# from torchmetrics import Accuracy
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import (LightningDataModule, LightningModule, Trainer, seed_everything)

# from torch_sparse import SparseTensor
# from torch_geometric.nn import SAGEConv, GATConv
# from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

dataset = MAG240MDataset(root = "/dbfs/mnt/ogb2022")

idx_dict = dataset.get_idx_split() # 'train', 'valid', 'test-whole', 'test-dev', 'test-challenge'
train_idx = idx_dict['train']
valid_idx = idx_dict['valid']
test_idx = idx_dict['test-challenge']

test_whole = idx_dict['test-whole']
test_dev = idx_dict['test-dev']
test_challenge = idx_dict['test-challenge']

def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:

    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

seed_idx = torch.cat([torch.from_numpy(train_idx), torch.from_numpy(valid_idx), torch.from_numpy(test_idx)])
seed_mask = index_to_mask(seed_idx, dataset.num_papers).to('cuda:0')

tvt_mapping = torch.full((dataset.num_papers,), -1, device = 'cuda:0')
tvt_mapping[seed_idx] = torch.arange(seed_idx.shape[0], device = 'cuda:0')

path = '/dbfs/mnt/ogb2022/mag240m_kddcup2021/fengtong/finetune/inputs'
metapath_path = '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/submission/results/bgrl_1024_72'
metapath_path_valid_full = '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/submission/results/labels_1024_72'

input_emb = [
    #(feat_name, input_dim, out_dim, path)
    ('x_degree', 2, 2, f'{path}/x_degree_tvt.pt'),
    ('x_m2v', 64, 64, f'{path}/x_m2v_tvt.npy'),
    ('x_rgat', 1024, 512, f'/dbfs/mnt/ogb2022/mag240m_kddcup2021/fengtong/finetune/inputs/rgat_epoch19_22102712_boot_hidden_1024.pt'),
    
    ('x_roberta', 768, 128, f'{path}/x_roberta_tvt.npy'),
    ('x_bgrl', 512, 512, f'{metapath_path}/train_val_test_hidden_512_k1.npy'),
    ('x_grace_1', 512, 512, f'{metapath_path}/train_val_test_hidden_512_grace_full_validation_k1.npy'),
    ('x_grace_0', 512, 512, f'{metapath_path}/train_val_test_hidden_512_grace_full_validation.npy'),
    ('x_grace_2', 512, 512, f'{metapath_path}/train_val_test_hidden_512_grace_full_validation_k2.npy'),
]

input_emb_metapath = [
    ('x_pcp_rw_lratio_valid_full', 153, 32, f'{metapath_path_valid_full}/x_pcp_rw_lratio_valid_full.npy'),
#     ('x_pcbp_rw_lratio_valid_full', 153, 32, f'{metapath_path_valid_full}/x_pcbp_rw_lratio_valid_full.npy'),
    ('x_pcpcbp_rw_lratio_valid_full', 153, 32, f'{metapath_path_valid_full}/x_pcpcbp_rw_lratio_valid_full.npy'),
#     ('x_pcbpcp_rw_lratio_valid_full', 153, 32, f'{metapath_path_valid_full}/x_pcbpcp_rw_lratio_valid_full.npy'),
    ('x_pcpcp_rw_lratio_valid_full', 153, 32, f'{metapath_path_valid_full}/x_pcpcp_rw_lratio_valid_full.npy'),
#     ('x_pcbpcbp_rw_lratio_valid_full', 153, 32, f'{metapath_path_valid_full}/x_pcbpcbp_rw_lratio_valid_full.npy'),
    ('x_pwbawp_rw_lratio_valid_full', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_lratio_valid_full.npy'),
    
    ('x_pcp_rw_lratio_valid_full_most_recent_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcp_rw_lratio_valid_full_most_recent_top_7_360.npy'),
    ('x_pcpcbp_rw_lratio_valid_full_most_recent_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcpcbp_rw_lratio_valid_full_most_recent_top_7_360.npy'),
#     ('x_pcpcp_rw_lratio_valid_full_most_recent_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcpcp_rw_lratio_valid_full_most_recent_top_7_360.npy'),
    ('x_pwbawp_rw_lratio_valid_full_most_recent_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_lratio_valid_full_most_recent_top_7_360.npy'),
    ('x_pcp_rw_lratio_valid_full_most_cited_top_10_360', 153, 32, f'{metapath_path_valid_full}/x_pcp_rw_lratio_valid_full_most_cited_top_10_360.npy'),
#     ('x_pcpcbp_rw_lratio_valid_full_most_cited_top_10_360', 153, 32, f'{metapath_path_valid_full}/x_pcpcbp_rw_lratio_valid_full_most_cited_top_10_360.npy'),
    ('x_pcpcp_rw_lratio_valid_full_most_cited_top_10_360', 153, 32, f'{metapath_path_valid_full}/x_pcpcp_rw_lratio_valid_full_most_cited_top_10_360.npy'),
    ('x_pwbawp_rw_lratio_valid_full_most_cited_top_10_360', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_lratio_valid_full_most_cited_top_10_360.npy'),


#     ('x_pcp_rw_lratio_valid_full_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcp_rw_lratio_valid_full_top_7_360_v2.npy'),
    ('x_pcpcbp_rw_lratio_valid_full_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcpcbp_rw_lratio_valid_full_top_7_360_v2.npy'),
#     ('x_pcbpcp_rw_lratio_valid_full_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcbpcp_rw_lratio_valid_full_top_7_360_v2.npy'),
    ('x_pcpcp_rw_lratio_valid_full_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcpcp_rw_lratio_valid_full_top_7_360_v2.npy'),
    
    ('x_pwbawp_rw_lratio_valid_full_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_lratio_valid_full_top_7_360_v2.npy'),
#     ('x_pcp_rw_lratio_valid_full_most_recent_top_7_360', 153, 32, f'{metapath_path_valid_full}/x_pcp_rw_lratio_valid_full_most_recent_top_7_360.npy'),
    ('x_pcpcbp_rw_c3_lratio_valid_full_v2', 153, 32, f'{metapath_path_valid_full}/x_pcpcbp_rw_c3_lratio_valid_full_v2.npy'),
  
    
#     ('x_pcbp_rw_lratio_valid_full_top_10', 153, 32, f'{metapath_path_valid_full}/x_pcbp_rw_lratio_valid_full_top_10.npy'),
#     ('x_pcpcbp_rw_lratio_valid_full_top_10', 153, 32, f'{metapath_path_valid_full}/x_pcpcbp_rw_lratio_valid_full_top_10.npy'),
#     ('x_pcbpcp_rw_lratio_valid_full_top_10', 153, 32, f'{metapath_path_valid_full}/x_pcbpcp_rw_lratio_valid_full_top_10.npy'),
#     ('x_pcpcp_rw_lratio_valid_full_top_10', 153, 32, f'{metapath_path_valid_full}/x_pcpcp_rw_lratio_valid_full_top_10.npy'),
#     ('x_pcbpcbp_rw_lratio_valid_full_top_10', 153, 32, f'{metapath_path_valid_full}/x_pcbpcbp_rw_lratio_valid_full_top_10.npy'),
#     ('x_pwbawp_rw_lratio_valid_full_top_10', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_lratio_valid_full_top_10.npy'),
    
    
    ('x_pwbawp_rw_c2_lratio_valid_full_v2', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_c2_lratio_valid_full_v2.npy'),
    ('x_pwbawp_rw_c4_lratio_valid_full_v2', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_c4_lratio_valid_full_v2.npy'),
    ('x_pwbawp_rw_l_lratio_valid_full_v2', 153, 32, f'{metapath_path_valid_full}/x_pwbawp_rw_l_lratio_valid_full_v2.npy'),
]

prefix = '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/submission'
rand_80_valid_mask = torch.load(f'{prefix}/results/valid_mask_1.pt')

class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int, dropout: float = 0.0, batch_norm: bool = True,
                 relu_last: bool = False, bias: bool = True):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, bias = bias))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels, bias = bias))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.dropout = dropout
        self.relu_last = relu_last

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_last:
                x = batch_norm(x).relu_()
            else:
                x = batch_norm(x.relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
class MyMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, hidden_size)
        self.l4 = torch.nn.Linear(hidden_size, hidden_size)
        self.l_output = torch.nn.Linear(hidden_size, num_classes)
        self.projection = torch.nn.Linear(input_dim, hidden_size)

        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size)
        self.bn4 = torch.nn.BatchNorm1d(hidden_size)

    def forward(self, x_input):
        """
        Feed forward network with residual connections.
        """
        x_proj = self.projection(x_input)
        x_ = self.bn1(F.leaky_relu(self.l1(x_input)))
        x = self.bn2(F.leaky_relu(self.l2(x_) + x_proj))
#         x = self.bn3(F.leaky_relu(self.l3(x) + x_proj))
        x = self.l_output(self.bn4(F.leaky_relu(self.l4(x) + x_)))
        return x    
      
class Model(torch.nn.Module):
    def __init__(self, in_channels: tuple, hidden_channels: tuple, out_channels: int, num_layers: int, num_metapath: int, device: None,
                 dropout: float = 0.0, batch_norm: bool = True, relu_last: bool = False):
        super(Model, self).__init__()

        self.mlps = ModuleList()
        self.input_dims = 0
        
        # non-metapath embs
        for i, j in zip(in_channels, hidden_channels):
            self.input_dims += j
            self.mlps.append(MLP(i, j, j, num_layers, dropout, batch_norm, relu_last))
            

        # metapath emb only - shared 
#         self.proj = torch.nn.Linear(self.input_dims, 64)
        self.mlp_metapath = MLP(153, 64, 64, num_layers, dropout, batch_norm, relu_last, bias = False)
        
        self.bias = torch.nn.Embedding.from_pretrained(torch.ones((num_metapath, 153)), freeze = False)
        self.metapath_idx = torch.arange(num_metapath, device = device)
        
#         self.input_dims += 153
#         self.conv = torch.nn.Conv1d(in_channels=num_metapath, out_channels=1, kernel_size=1, stride=1)

#         self.input_dims += 153
#         self.relation_attns = Linear(64 * 3, 1)
#         self.relation_norms = BatchNorm1d(64*3)

        
#         self.mlp = MLP(self.input_dims, 512, out_channels, num_layers, dropout, batch_norm, relu_last)
        self.mlp = MyMLP(self.input_dims + 64*3, 512, out_channels)
        self.device = device

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()
        self.reset_parameters()

    def forward(self, xs, xs_metapath):
        outs = []
        for x, mlp in zip(xs, self.mlps):
            outs.append(mlp(x))
            
        metapath_out_max = None
        metapath_out_min = None
        metapath_out_sum = None
        for i, x in enumerate(xs_metapath):
            if self.training:
                x = (x + torch.normal(0, 3e-6, size=(x.shape[0], x.shape[1])).to(self.device))
                
            if metapath_out_max is None:
                metapath_out_max = self.mlp_metapath(x * self.bias(self.metapath_idx[i]))
            else:
                metapath_out_max = torch.maximum(metapath_out_max, self.mlp_metapath(x * self.bias(self.metapath_idx[i])))
                
            if metapath_out_min is None:
                metapath_out_min = self.mlp_metapath(x * self.bias(self.metapath_idx[i]))
            else:
                metapath_out_min = torch.minimum(metapath_out_min, self.mlp_metapath(x * self.bias(self.metapath_idx[i])))
            
            if metapath_out_sum is None:
                metapath_out_sum = self.mlp_metapath(x * self.bias(self.metapath_idx[i]))
            else:
                metapath_out_sum = metapath_out_sum + self.mlp_metapath(x * self.bias(self.metapath_idx[i]))
            
#             if metapath_out_max is None:
#                 metapath_out_max = x * self.bias(self.metapath_idx[i])
#             else:
#                 metapath_out_max = torch.maximum(metapath_out_max, x * self.bias(self.metapath_idx[i]))
                
#             if metapath_out_min is None:
#                 metapath_out_min = x * self.bias(self.metapath_idx[i])
#             else:
#                 metapath_out_min = torch.minimum(metapath_out_min, x * self.bias(self.metapath_idx[i]))
            
#             if metapath_out_sum is None:
#                 metapath_out_sum = x * self.bias(self.metapath_idx[i])
#             else:
#                 metapath_out_sum = metapath_out_sum + x * self.bias(self.metapath_idx[i])
            
#         metapath_out_max = self.mlp_metapath(metapath_out_max)
#         metapath_out_min = self.mlp_metapath(metapath_out_min)
#         metapath_out_sum = self.mlp_metapath(metapath_out_sum)

#         metapath_out = torch.stack(xs_metapath, dim=1) # torch.Size([10240, 10, 153])
#         metapath_out = self.conv(metapath_out) # torch.Size([10240, 1, 144])
#         metapath_out = torch.squeeze(metapath_out, dim=1) # torch.Size([10240, 144])

#         out_feat = torch.stack(xs_metapath, dim=1)
#         out_attn = self.relation_attns(out_feat)
#         out_attn = F.softmax(out_attn, dim=1)
#         out_attn = torch.permute(out_attn, (0,2,1))
#         out = torch.bmm(out_attn, out_feat)[:, 0]
#         metapath_out = self.relation_norms(out)

#         metapath_out = torch.cat([metapath_out_max, metapath_out_min, metapath_out_sum], dim=1)
#         out_attn = self.relation_attns(metapath_out)
#         out_attn = F.softmax(out_attn, dim=1)
        
# #         out = torch.bmm(out_attn, out_feat)[:, 0]
#         out = torch.mul(out_attn.expand(metapath_out.shape[0], 64*3), metapath_out)
#         metapath_out = self.relation_norms(out)

        x = torch.cat(outs, dim=-1)
#         y = self.proj(x), y * metapath_out_max, y * metapath_out_min, y * metapath_out_sum
        x = torch.cat([x, metapath_out_max, metapath_out_min, metapath_out_sum], dim=1).relu_()
#         x = torch.cat([x, metapath_out], dim=1).relu_()
        
        return self.mlp(x)


def label_distr(year, data_df):
    df = data_df[data_df['year'] == year]['label'].value_counts().reset_index()
    df.columns = ['year', 'count']
    df = df.sort_values('year')
    
    d = {}
    for year, count in zip(df['year'], df['count']):
        d[year] = count
    res = []
    for i in range(153):
        res.append(d[i]) if i in d else res.append(0)
    return res

def label_dist_emb_create():
    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test-whole')

    all_idx = np.concatenate([train_idx, valid_idx, test_idx])
    all_type = np.concatenate([["train"] * train_idx.shape[0], ["valid"] * valid_idx.shape[0], ["test"] * test_idx.shape[0]])

    data_df = pd.DataFrame(all_idx)
    data_df.columns = ['idx']
    data_df['year'] = dataset.all_paper_year[all_idx]
    data_df['label'] = dataset.paper_label[all_idx].astype(int)
    data_df['type'] = all_type

    del all_idx
    del all_type

    label_dist_arr = np.zeros((14, 153))
    for year in range(2009, 2022):
        x = label_distr(year, data_df)
        arr = np.array(x + np.ones((len(x))), dtype=np.int32)
        label_dist_arr[year - 2009 + 1] = np.array(arr/np.sum(arr), dtype=np.float16)

    label_dist_emb = torch.nn.Embedding.from_pretrained(torch.tensor(label_dist_arr), freeze=True)
    del label_dist_arr
    del data_df
    return label_dist_emb

label_dist_emb = label_dist_emb_create()


def load_data(input_emb, device=None, replace=None):

    x_all = []
    for i, (feat_name, _, _, path) in enumerate(input_emb):
        print(f'Loading {i} data in {path}')
        if feat_name in ['x_bgrl', 'x_rgat', 'x_rgat_2', 'x_degree', 'x_grace_0', 'x_grace_1', 'x_grace_2']:
            feat = torch.load(path)
        else:
            feat = torch.from_numpy(np.load(path))
            
        if feat_name in ['x_degree']:
            feat = torch.log(feat.float() + 1).half()

        if replace is not None:
            assert replace in ['zero', 'prev_year']
            replace_mask = torch.std(feat, dim=1) < 1e-5
            if replace == 'zero':
                feat[replace_mask] = torch.zeros((replace_mask.sum(), 153)) 
            elif replace == 'prev_year':
                years = dataset.all_paper_year[seed_idx_whole[replace_mask]]
                years[years < 2009] = 2009
                feat[replace_mask] = label_dist_emb(torch.from_numpy(years - 2009)).float()
            
        x_all.append(feat)

    return x_all

y_all = torch.from_numpy(dataset.all_paper_label).long()

x_all_metapath = load_data(input_emb_metapath, replace='prev_year')

def train(input_emb, epoch, metapath_indexes, x_all):
    model.train()
    total_examples = total_loss = total_correct = 0
    
    with tqdm(total=len(train_loader), desc='(T)') as pbar:
        for idx in train_loader:
            optimizer.zero_grad()
            y_hat = model([x[tvt_mapping[idx]].to(device = args.device, dtype = torch.float32) for x in x_all],
                          [x[tvt_mapping[idx]].to(device = args.device, dtype = torch.float32) for i, x in enumerate(x_all_metapath) if i in metapath_indexes])
            y_true = y_all[idx].to('cuda')
            loss = F.cross_entropy(y_hat, y_true)

            loss.backward()
            optimizer.step()
            
            total_correct += int((y_hat.argmax(dim=-1) == y_true).sum())

            batch_size = idx.shape[0]
            total_examples += batch_size
            total_loss += float(loss) * batch_size

            pbar.set_postfix({'epoch':epoch, 'loss': total_loss / total_examples, 'acc': total_correct / total_examples})
            pbar.update()
        scheduler.step()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test(loader, x_all, metapath_indexes):
    model.eval()
    total_examples = total_correct = 0
    res = []
    for idx in loader:
        y_hat = model([x[tvt_mapping[idx]].to(device = args.device, dtype = torch.float32) for x in x_all],
                          [x[tvt_mapping[idx]].to(device = args.device, dtype = torch.float32) for i, x in enumerate(x_all_metapath) if i in metapath_indexes])
        y_true = y_all[idx].to('cuda')

        total_correct += int((y_hat.argmax(dim=-1) == y_true).sum())
        total_examples += idx.shape[0]
        
        res.append(y_hat)

    return total_correct / total_examples, torch.cat(res, dim=0)

@torch.no_grad()
def test_prediction(loader, x_all, metapath_indexes):
    model.eval()
    evaluator = MAG240MEvaluator()
    y_preds = []
    for idx in loader:
        y_hat = model([x[tvt_mapping[idx]].to(device = args.device, dtype = torch.float32) for x in x_all],
                          [x[tvt_mapping[idx]].to(device = args.device, dtype = torch.float32) for i, x in enumerate(x_all_metapath) if i in metapath_indexes])
        y_preds.append(y_hat.detach().cpu())
    return torch.cat(y_preds, dim=0)

train_idx = train_idx[(dataset.all_paper_year[train_idx] >= 2017) & (dataset.paper_label[train_idx].astype(int) != 100)]
train_idx = torch.from_numpy(train_idx)
print("train_idx size: ", train_idx.shape)

valid_idx = torch.from_numpy(valid_idx)
test_idx = torch.from_numpy(test_idx)

train_idx.share_memory_()
valid_idx.share_memory_()
test_idx.share_memory_()

class args:
    hidden_channels = 1536
    num_layers = 2
    out_channels = 153
    dropout=0.5
    device=0
    seed=42
    batch_size = 1024 * 10
    
    lr = 0.001
    weight_decay = 0
    step_size = 5
    gamma = 0.2
    
    def __str__(self):
        return f'hidden_channel={self.hidden_channels}, num_layer={self.num_layers}, dropout={self.dropout}, weight_decay={self.weight_decay}, lr={self.lr}, step_size={self.step_size}, gamma={self.gamma}'

state = True

input_emb_fix = [
    ('x_degree', 2, 2, f'{path}/x_degree_tvt.pt'),
]
if state:
    input_emb_random = [
        ('x_roberta', 768, 128, f'{path}/x_roberta_tvt.npy'),
        ('x_bgrl', 512, 512, f'{metapath_path}/train_val_test_hidden_512_k1_challange.npy'),
        ('x_grace_1', 512, 512, f'{metapath_path}/train_val_test_hidden_512_grace_full_validation_k1.npy'),
        ('x_rgat', 1024, 512, f'/dbfs/mnt/ogb2022/mag240m_kddcup2021/fengtong/finetune/inputs/rgat_epoch19_22102712_boot_hidden_1024.pt'),
        ('x_m2v', 64, 64, f'{path}/x_m2v_tvt.npy'),
    ]
    x_all_loaded = load_data([input_emb_fix[0], input_emb_random[0], input_emb_random[1], input_emb_random[2], input_emb_random[3], input_emb_random[4]])
else:
    input_emb_random = [
        ('x_roberta', 768, 128, f'{path}/x_roberta_tvt.npy'),
        ('x_bgrl', 512, 512, f'{metapath_path}/train_val_test_hidden_512_k1_challange.npy'),
        ('x_grace_1', 512, 512, f'{metapath_path}/train_val_test_hidden_512_grace_full_validation_k1.npy'),
        ('x_rgat', 1024, 512, f'/dbfs/mnt/ogb2022/mag240m_kddcup2021/fengtong/finetune/inputs/rgat_epoch19_22102712_boot_hidden_1024.pt'),
        ('x_m2v', 64, 64, f'{path}/x_m2v_tvt.npy'),
        ('x_grace_0', 512, 512, f'{metapath_path}/train_val_test_hidden_512_grace_full_validation.npy'),
        ('x_grace_2', 512, 512, f'{metapath_path}/train_val_test_hidden_512_grace_full_validation_k2.npy'),
    ]
    x_all_loaded = load_data([input_emb_fix[0], input_emb_random[0], input_emb_random[1], input_emb_random[2], input_emb_random[3], input_emb_random[4], input_emb_random[5], input_emb_random[6]])

    
for k in range(len(input_emb_random)):
    input_emb_random[k] = (input_emb_random[k][0], int(input_emb_random[k][1]*.9), input_emb_random[k][2], input_emb_random[k][3])

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

if state:
    train_loader = DataLoader(torch.cat([train_idx, valid_idx[rand_80_valid_mask]], dim=0), args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_idx[~rand_80_valid_mask], args.batch_size, shuffle=False)
else:
    train_loader = DataLoader(torch.cat([train_idx, valid_idx], dim=0), args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_idx, args.batch_size, shuffle=False)

test_loader = DataLoader(test_idx, args.batch_size, shuffle=False)
    
epoches = 15
train_acc_res = []
valid_acc_res = []
y_valid_hat_final = None
y_test_hat_final = None
best_valid_accs = []

for ii in range(10):
    
    np.random.seed(ii)
    torch.manual_seed(ii)
    torch.cuda.manual_seed_all(ii)
    
#     perm = torch.randperm(len(x_all_metapath))
    metapath_indexes = torch.arange(len(x_all_metapath)) # perm[:len(x_all_metapath)//3]
#     input_emb_new = [input_emb_fix[0], input_emb_fix[1], input_emb_fix[2], input_emb_random[ii % len(input_emb_random)]]
    if state:
        input_emb_new = [
            input_emb_fix[0],
            input_emb_random[0],
            input_emb_random[1],
            input_emb_random[2],
            input_emb_random[3],
            input_emb_random[4],
        ]
    else:
        input_emb_new = [
            input_emb_fix[0],
            input_emb_random[0],
            input_emb_random[1],
            input_emb_random[2],
            input_emb_random[3],
            input_emb_random[4],
            input_emb_random[5],
            input_emb_random[6],
        ]
#     x_all = load_data(input_emb_new, args.device) 
    if state:
        x_all = [
            x_all_loaded[0], 
            x_all_loaded[1][:, torch.randperm(768)[:int(768*.9)]],
            x_all_loaded[2][:, torch.randperm(512)[:int(512*.9)]],
            x_all_loaded[3][:, torch.randperm(512)[:int(512*.9)]],
            x_all_loaded[4][:, torch.randperm(1024)[:int(1024*.9)]],
            x_all_loaded[5][:, torch.randperm(64)[:int(64*.9)]],
        ]
    else:
        x_all = [
            x_all_loaded[0], 
            x_all_loaded[1][:, torch.randperm(768)[:int(768*.9)]],
            x_all_loaded[2][:, torch.randperm(512)[:int(512*.9)]],
            x_all_loaded[3][:, torch.randperm(512)[:int(512*.9)]],
            x_all_loaded[4][:, torch.randperm(1024)[:int(1024*.9)]],
            x_all_loaded[5][:, torch.randperm(64)[:int(64*.9)]],
            x_all_loaded[6][:, torch.randperm(512)[:int(512*.9)]],
            x_all_loaded[7][:, torch.randperm(512)[:int(512*.9)]],
        ]
    
    model = Model(in_channels=[f[1] for f in input_emb_new], hidden_channels=[f[2] for f in input_emb_new], 
                device = args.device,
                out_channels=dataset.num_classes, 
                num_layers=args.num_layers, 
                dropout=args.dropout, 
                num_metapath=len(x_all_metapath), 
                relu_last=True)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    best_valid_acc = 0
    best_y_valid_hat = None
    best_y_test_hat = None
    for epoch in range(1, epoches+1):
        loss, train_acc = train(input_emb_new, epoch, metapath_indexes, x_all)
        valid_acc, y_valid_hat = test(valid_loader, x_all, metapath_indexes)
        
        if valid_acc > best_valid_acc:
            y_test_hat = test_prediction(test_loader, x_all, metapath_indexes)
            best_valid_acc = valid_acc
            best_y_valid_hat = y_valid_hat
            best_y_test_hat = y_test_hat

        train_acc_res.append(train_acc)
        valid_acc_res.append(valid_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')
    
    best_valid_accs.append(best_valid_acc)
    y_valid_hat_final = best_y_valid_hat if y_valid_hat_final is None else best_y_valid_hat + y_valid_hat_final
    y_test_hat_final = best_y_test_hat if y_test_hat_final is None else best_y_test_hat + y_test_hat_final
    
print(best_valid_accs)    
if state:
    acc_final = accuracy_score(y_all[valid_idx[~rand_80_valid_mask]].cpu().numpy(), y_valid_hat_final.argmax(dim = -1).cpu().numpy())
else:
    acc_final = accuracy_score(y_all[valid_idx].cpu().numpy(), y_valid_hat_final.argmax(dim = -1).cpu().numpy())
print(acc_final)

if not state:
    model_version = 'final_bagging_20221101_22_mehdi_' + str(acc_final)
    res = {'y_pred': y_test_hat_final.argmax(dim = -1).cpu()}
    evaluator = MAG240MEvaluator()
    evaluator.save_test_submission(res, "/databricks/driver/", mode = 'test-challenge')
    # !cp '/databricks/driver/y_pred_mag240m_test-challenge.npz' '{prefix}/y_pred_mag240m_test-challenge__{model_version}.npz'
    print(f'Find the test submission file at: {prefix}/y_pred_mag240m_test-challenge__{model_version}.npz')
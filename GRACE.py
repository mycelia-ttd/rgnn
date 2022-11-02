from torch_geometric.nn import HeteroConv, Linear, SAGEConv, GATv2Conv
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

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
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (
    LightningDataModule, LightningModule, Trainer, seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler
import math

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch.nn import MultiheadAttention
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T
from GCL.models.contrast_model import WithinEmbedContrast
from GCL.models import DualBranchContrast

ROOT = "/dbfs/mnt/ogb2022"
dataset = MAG240MDataset(root=ROOT)


class args:
    hidden_channels = 512
    batch_size = 512
    dropout = 0.42
    epochs = 3
    model = 'rgat'
    sizes = [22, 15]
    device = 'cuda'
    evaluate = False

    num_paper_features = 128 + 64
    num_features = 128 + 64 + 64 + 16
    num_relations = 5
    num_classes = 153


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]
    year_id: Tensor
    n_id: Tensor
    y_full: Tensor
    batch_size: int

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
            year_id=self.year_id.to(*args, **kwargs),
            n_id=self.n_id.to(*args, **kwargs),
            y_full=self.y_full.to(*args, **kwargs),
            batch_size=self.batch_size
        )


class RGNN(torch.nn.Module):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 num_paper_features: int,
                 label_dist_emb: None, pe_embedding: None,
                 heads: int = 4, dropout: float = 0.5, weight_decay: float = 1e-5):
        super().__init__()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.hidden_channels = hidden_channels

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        self.aug1 = A.Compose(
            [A.EdgeRemoving(pe=0.33), A.FeatureMasking(pf=0.1)])
        self.aug2 = A.Compose(
            [A.EdgeRemoving(pe=0.33), A.FeatureMasking(pf=0.1)])
        self.contrast_model = WithinEmbedContrast(loss=L.BarlowTwins())

        self.convs.append(
            ModuleList([
                GATv2Conv(in_channels, hidden_channels // heads, heads,
                          add_self_loops=False) for _ in range(num_relations)
            ]))

        for _ in range(num_layers - 1):
            self.convs.append(
                ModuleList([
                    GATv2Conv(hidden_channels, hidden_channels // heads,
                              heads, add_self_loops=False)
                    for _ in range(num_relations)
                ]))

        for _ in range(num_layers):
            self.norms.append(
                ModuleList([
                    BatchNorm1d(hidden_channels) for _ in range(num_relations)
                ])
            )

        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.label_dist_emb = label_dist_emb
        self.label_dist_emb_mlp = Sequential(
            Linear(153, 64),
            BatchNorm1d(64),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
        )
        self.min_year = 2009

        self.pe_embedding = pe_embedding

        self.label_emb = torch.nn.Embedding(153, num_paper_features)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        proj_dim = 256
        self.fc1 = torch.nn.Linear(hidden_channels, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_channels)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def forward(self, batch, predict=False) -> Tensor:

        x = batch.x
        adjs_t = batch.adjs_t
        year_id = batch.year_id

        year_input = year_id - self.min_year
        year_input = torch.where(year_input < 0, 0, year_input)

        year_emb = self.label_dist_emb(year_input).float()
        year_emb = self.label_dist_emb_mlp(year_emb)

        batch_label_mask = batch.y_full >= 0
        paper_nid_mask = batch.n_id < num_papers
        batch_label_mask_idx = torch.arange(batch.n_id.shape[0], dtype=torch.long, device=batch_label_mask.device)[
            paper_nid_mask][valid_mask[batch.n_id[paper_nid_mask]]]
        batch_label_mask[batch_label_mask_idx] = False
        batch_label_mask[:batch.batch_size] = False

        if self.training:
            more_rand_mask = torch.rand(batch_label_mask.size(
                0), device=batch_label_mask.device) < 0.02
            batch_label_mask = batch_label_mask & (~more_rand_mask)

            masked_label_emb = self.label_emb(torch.randint(
                0, 153, ((more_rand_mask).sum(),), device=batch_label_mask.device))
            masked_label_index = torch.arange(
                batch.x.shape[0], dtype=torch.int32, device=batch_label_mask.device)[more_rand_mask]
            x = x.index_add(dim=0, index=masked_label_index,
                            source=masked_label_emb)

        masked_label_emb = self.label_emb(batch.y_full[batch_label_mask])
        masked_label_index = torch.arange(
            batch.x.shape[0], dtype=torch.int32, device=batch_label_mask.device)[batch_label_mask]
        x = x.index_add(dim=0, index=masked_label_index,
                        source=masked_label_emb)

        x = torch.cat([x, year_emb, self.pe_embedding(year_input)], 1)
        # twin outputs
        x1 = x.clone()
        x2 = x.clone()

        for i, adj_t in enumerate(adjs_t):
            row, col, edge_attr = adj_t.t().coo()
            edge_index_t = torch.stack([row, col], dim=0)
            adj_t_size = adj_t.size(0)

            x_target = x[:adj_t_size]
            x_target1 = x1[:adj_t_size]
            x_target2 = x2[:adj_t_size]

            # todo: check no corruption from augmentation in place
            x1_t, edge_index1, edge_attr1 = self.aug1(
                x_target1, edge_index_t, edge_weight=edge_attr)
            x2_t, edge_index2, edge_attr2 = self.aug2(
                x_target2, edge_index_t, edge_weight=edge_attr)

            # X1
            out = self.skips[i](x1_t)
            for j in range(self.num_relations):
                if (edge_attr1 == j).sum() > 0:
                    out_tmp = self.convs[i][j](
                        (x1, x1_t), edge_index1[:, edge_attr1 == j])
                    out += self.norms[i][j](out_tmp)
            x1 = out
            x1 = F.elu(x1)

            # X2
            out = self.skips[i](x2_t)
            for j in range(self.num_relations):
                if (edge_attr2 == j).sum() > 0:
                    out_tmp = self.convs[i][j](
                        (x2, x2_t), edge_index2[:, edge_attr2 == j])
                    out += self.norms[i][j](out_tmp)
            x2 = out
            x2 = F.elu(x2)

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                subadj_t = subadj_t.set_value(None, layout=None)
                if subadj_t.nnz() > 0:
                    x_conv = self.convs[i][j]((x, x_target), subadj_t)
                    out += self.norms[i][j](x_conv)

            x = F.elu(out)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if predict:
            return x

        return self.mlp(x), x1, x2


def convert_batch(batch_size, n_id, adjs):
    x = full_feat_pai_m2v[n_id].to(torch.float)
    y = expand_y[n_id[:batch_size]].to(torch.long)

    y_full = torch.full((n_id.shape[0],), -1)
    paper_n_id_mask = n_id < num_papers
    y_full[paper_n_id_mask] = expand_y[n_id[paper_n_id_mask]]

    year_id = torch.from_numpy(all_paper_year[n_id])
    return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs], year_id=year_id, n_id=n_id, y_full=y_full, batch_size=batch_size)


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
    all_type = np.concatenate([["train"] * train_idx.shape[0],
                              ["valid"] * valid_idx.shape[0], ["test"] * test_idx.shape[0]])

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
        label_dist_arr[year - 2009 +
                       1] = np.array(arr/np.sum(arr), dtype=np.float16)

    label_dist_emb = torch.nn.Embedding.from_pretrained(
        torch.tensor(label_dist_arr), freeze=True)
    del label_dist_arr
    del data_df
    return label_dist_emb


def create_pe_embedding():
    max_len = 25
    d_model = 16  # 153+128+64
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    print(pe.shape)

    return torch.nn.Embedding.from_pretrained(pe, freeze=True)


if __name__ == "__main__":
    num_papers = dataset.num_papers
    num_authors = dataset.num_authors
    num_institutions = dataset.num_institutions

    full_feat_pai_m2v = torch.from_numpy(np.load(
        '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/full_feat_pai_m2v.npy'))
    adj_t = torch.load('/dbfs/mnt/ogb2022/mag240m_kddcup2021/full_adj_t.pt')

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test-whole')

    expand_y = torch.from_numpy(dataset.all_paper_label.astype(int))
    all_paper_year = np.concatenate((dataset.all_paper_year, np.full(
        (num_authors, ), -1), np.full((dataset.num_institutions, ), -1)))

    label_dist_emb = label_dist_emb_create()
    pe_embedding = create_pe_embedding()

    train_idx = dataset.get_idx_split('train')
    train_idx = train_idx[(dataset.all_paper_year[train_idx] >= 2012) & (
        dataset.paper_label[train_idx].astype(int) != 100)]
    train_idx = torch.from_numpy(train_idx)
    val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    test_idx = torch.from_numpy(dataset.get_idx_split('test-challenge'))

    train_idx.share_memory_()
    val_idx.share_memory_()
    test_idx.share_memory_()

    valid_mask = index_to_mask(val_idx, size=num_papers)

    train_loader = NeighborSampler(adj_t, node_idx=torch.cat([train_idx, val_idx, test_idx]),  # val_idx
                                   sizes=args.sizes, return_e_id=False,
                                   transform=convert_batch,
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=4)

    val_loader = NeighborSampler(adj_t, node_idx=val_idx[~rand_80_valid_mask],
                                 sizes=args.sizes, return_e_id=False,
                                 transform=convert_batch,
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=4)

    model = RGNN(model=args.model, in_channels=args.num_features,
                 out_channels=args.num_classes, hidden_channels=args.hidden_channels,
                 num_relations=args.num_relations, num_layers=len(args.sizes),
                 label_dist_emb=label_dist_emb, pe_embedding=pe_embedding, dropout=args.dropout,
                 num_paper_features=args.num_paper_features)

    scaler = torch.cuda.amp.GradScaler()
    model = model.to(args.device)
    print(f'#Params {sum([p.numel() for p in model.parameters()])}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00142)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.4)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(
        tau=0.2), mode='L2L', intraview_negs=True).to(args.device)

    def train(train_loader, post_year=2012):
        model.train()
        total_examples = total_loss = total_correct = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch = batch.to(args.device)
                y_hat, z1, z2 = model(batch, gmix=False)

                h1, h2 = [model.project(x) for x in [z1, z2]]
                train_loss = contrast_model(h1, h2)

                # train labels only
                cross_entropy_mask = batch.year_id[:batch.batch_size] >= post_year
                cross_entropy_mask = cross_entropy_mask & (
                    batch.y >= 0)  # only on known labels
                cross_entropy_mask = cross_entropy_mask & (
                    ~valid_mask[batch.n_id[:batch.batch_size].to('cpu')].to('cuda'))

                mask_size = cross_entropy_mask.sum()
                if mask_size > 0:
                train_loss += F.cross_entropy(
                    y_hat[cross_entropy_mask],
                    batch.y[cross_entropy_mask]
                )

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_correct += int((y_hat[cross_entropy_mask].argmax(dim=-1)
                                 == batch.y[cross_entropy_mask]).sum().cpu())

            total_examples += cross_entropy_mask.sum().cpu()
            total_loss += float(train_loss) * batch.batch_size
        scheduler.step()

        return total_loss / total_examples, total_correct / total_examples

    @torch.no_grad()
    def test(loader):
        model.eval()
        total_examples = total_correct = 0
        for batch in tqdm(loader):
            batch = batch.to(args.device)
            with torch.cuda.amp.autocast():
                y_hat, _, _ = model(batch, gmix=False)

            total_correct += int((y_hat.argmax(dim=-1) == batch.y).sum().cpu())
            total_examples += batch.batch_size

        return total_correct / total_examples

    def train_test(num_epochs):
        train_acc_res = []
        valid_acc_res = []

        for epoch in range(1, num_epochs+1):
            loss, train_acc = train(train_loader)
            valid_acc = test(val_loader)

            train_acc_res.append(train_acc)
            valid_acc_res.append(valid_acc)
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

        return train_acc_res, valid_acc_res

    train_test(15)

    pred_loader = NeighborSampler(adj_t, node_idx=torch.cat([train_idx, val_idx, test_idx]),
                                  sizes=args.sizes, return_e_id=False,
                                  transform=convert_batch,
                                  batch_size=512, shuffle=False,
                                  num_workers=6)

    @torch.no_grad()
    def generate_hidden_embeddings(loader):
        model.eval()
        total_examples = total_correct = 0
        res_inner = []
        for batch in tqdm(loader):
        batch = batch.to(args.device)
        with torch.cuda.amp.autocast():
            h = model(batch, gmix=False, predict=True)
            res_inner.append(h.detach().cpu())

        res_inner = torch.cat(res_inner, dim=0)
        return res_inner

    res = generate_hidden_embeddings(pred_loader)

    prefix = '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/submission'
    model_name = "bgrl_1024_72"

    torch.save(res, f'{prefix}/results/{model_name}/hidden_features.npy')

import numpy as np
import pandas as pd
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import os
import time
import glob
import argparse
import copy
import os.path as osp
from tqdm import tqdm
import math

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import StepLR
import GCL.losses as L
from GCL.models import BootstrapContrast

from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (
    LightningDataModule, LightningModule, Trainer, seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch_geometric.utils import index_to_mask
from torch_geometric.nn import HeteroConv, Linear, SAGEConv, GATv2Conv

import GCL.losses as L
from torch.optim.lr_scheduler import StepLR
from GCL.models import BootstrapContrast
import GCL.augmentors as A
import torch_geometric.transforms as T
from GCL.models.contrast_model import WithinEmbedContrast
from GCL.models import DualBranchContrast

ROOT = "/dbfs/mnt/ogb2022"
dataset = MAG240MDataset(root=ROOT)


class args:
    hidden_channels = 512
    batch_size = 256
    dropout = 0.42
    epochs = 3
    model = 'rgat'
    sizes = [25, 15]
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


def convert_batch(batch_size, n_id, adjs):
    x = full_feat_pai_m2v[n_id].to(torch.float)
    y = expand_y[n_id[:batch_size]].to(torch.long)

    y_full = torch.full((n_id.shape[0],), -1)
    paper_n_id_mask = n_id < num_papers
    y_full[paper_n_id_mask] = expand_y[n_id[paper_n_id_mask]]

    year_id = torch.from_numpy(all_paper_year[n_id])
    return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs], year_id=year_id, n_id=n_id, y_full=y_full, batch_size=batch_size)


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

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, adjs_t) -> Tensor:

        curr_target = None
        for i, (x, edge_index, edge_attrs, adj_t_size) in enumerate(adjs_t):
            if curr_target is not None:
                x = curr_target

            x_target = x[:adj_t_size]
            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                if (edge_attrs == j).sum() > 0:
                    out_tmp = self.convs[i][j](
                        (x, x_target), edge_index[:, edge_attrs == j])
                    out += self.norms[i][j](out_tmp)
            x = out
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(x)
            curr_target = x

        return x, self.projection_head(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, dropout, num_paper_features):

        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.dropout = dropout
        self.num_papers = num_papers

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

        self.aug1 = A.Compose(
            [A.EdgeRemoving(pe=0.33), A.FeatureMasking(pf=0.1)])
        self.aug2 = A.Compose(
            [A.EdgeRemoving(pe=0.33), A.FeatureMasking(pf=0.1)])

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_channels, args.hidden_channels),
            BatchNorm1d(args.hidden_channels),
            torch.nn.PReLU(),
            torch.nn.Dropout(self.dropout)
        )

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def sparse_to_dense_adj(self, adj_t, x):

        row, col, edge_attr = adj_t.t().coo()
        edge_index_t = torch.stack([row, col], dim=0)
        adj_t_size = adj_t.size(0)

        return x, edge_index_t, edge_attr, adj_t_size

    def forward(self, batch, validation=False):
        x = batch.x
        adjs_t = batch.adjs_t
        year_id = batch.year_id

        year_input = year_id - self.min_year
        year_input = torch.where(year_input < 0, 0, year_input)

        year_emb = self.label_dist_emb(year_input).float()
        year_emb = self.label_dist_emb_mlp(year_emb)

        batch_label_mask = batch.y_full >= 0
        valid_set = torch.arange(
            batch.x.shape[0], dtype=torch.long, device=batch_label_mask.device)
        paper_ids = batch.n_id[batch.n_id < self.num_papers]
        valid_paper_mask = valid_mask[paper_ids]
        batch_label_mask[valid_set[batch.n_id <
                                   self.num_papers][valid_paper_mask]] = False
        batch_label_mask[:batch.batch_size] = False

        if self.training:  # replace 1/25 of labels with random ones
            more_rand_mask = torch.rand(batch_label_mask.size(
                0), device=batch_label_mask.device) < 0.01
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

        adjs_t = [self.sparse_to_dense_adj(a, x) for a in adjs_t]
        if validation:
            return self.online_encoder(adjs_t)[0]

        aug1_adjs_t = [list(self.aug1(a, b, c)) + [s]
                       for (a, b, c, s) in adjs_t]
        aug2_adjs_t = [list(self.aug2(a, b, c)) + [s]
                       for (a, b, c, s) in adjs_t]

        h1, h1_online = self.online_encoder(aug1_adjs_t)
        h2, h2_online = self.online_encoder(aug2_adjs_t)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(aug1_adjs_t)
            _, h2_target = self.get_target_encoder()(aug2_adjs_t)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


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
    max_len = 20
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

    # concatenated whitened Roberta + metapath2vec
    full_feat_pai_m2v = torch.from_numpy(np.load(
        '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/full_feat_pai_m2v.npy'))

    adj_t = torch.load('/dbfs/mnt/ogb2022/mag240m_kddcup2021/full_adj_t.pt')

    all_paper_year = np.concatenate((dataset.all_paper_year, np.full(
        (num_authors, ), -1), np.full((num_institutions, ), -1)))

    # remove irrelvalent training samples
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))

    train_idx = train_idx[(dataset.all_paper_year[train_idx] > 2009) & (
        dataset.paper_label[train_idx].astype(int) != 100)]
    val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    test_idx = torch.from_numpy(dataset.get_idx_split('test-whole'))

    train_idx.share_memory_()
    val_idx.share_memory_()
    test_idx.share_memory_()

    valid_mask = index_to_mask(val_idx, size=num_papers)
    expand_y = torch.from_numpy(dataset.all_paper_label.astype(int))

    label_dist_emb = label_dist_emb_create()
    pe_embedding = create_pe_embedding()

    train_loader = NeighborSampler(adj_t, node_idx=torch.cat([train_idx, val_idx, test_idx]),
                                   sizes=args.sizes, return_e_id=False,
                                   transform=convert_batch,
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=6)

    val_loader = NeighborSampler(adj_t, node_idx=val_idx,
                                 sizes=args.sizes, return_e_id=False,
                                 transform=convert_batch,
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=4)

    seed_everything(3)

    mlp_projection = Sequential(
        Linear(args.hidden_channels, args.hidden_channels),
        BatchNorm1d(args.hidden_channels),
        ReLU(inplace=True),
        Dropout(p=args.dropout),
        Linear(args.hidden_channels, args.num_classes),
    )
    mlp_projection.to(args.device)

    model = RGNN(model=args.model, in_channels=args.num_features,
                 out_channels=args.num_classes, hidden_channels=args.hidden_channels,
                 num_relations=args.num_relations, num_layers=len(args.sizes),
                 label_dist_emb=label_dist_emb, pe_embedding=pe_embedding, dropout=args.dropout,
                 num_paper_features=args.num_paper_features)

    print(f'#Params {sum([p.numel() for p in model.parameters()])}')

    scaler = torch.cuda.amp.GradScaler()
    encoder = Encoder(model, args.dropout, args.num_paper_features)
    encoder.to(args.device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(mlp_projection.parameters()), lr=0.00142)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.4)
    contrast_model = BootstrapContrast(
        loss=L.BootstrapLatent(), mode='L2L').to('cuda')

    def train(epoch):
        encoder.train()
        mlp_projection.train()
        total_examples = total_loss = total_correct = 0
        with tqdm(total=len(train_loader), desc='(T)') as pbar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    batch = batch.to(args.device)
                    h1, h2, h1_pred, h2_pred, h1_target, h2_target = encoder(
                        batch)

                    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(
                    ), h2_target=h2_target.detach())

                    y_hat = mlp_projection((h1 + h2) / 2.0)

                    cross_entropy_mask = (batch.y >= 0)
                    cross_entropy_mask = cross_entropy_mask & (
                        ~valid_mask[batch.n_id[:batch.batch_size].to('cpu')].to('cuda'))

                    if cross_entropy_mask.sum() > 0:
                        loss += F.cross_entropy(y_hat[cross_entropy_mask],
                                                batch.y[cross_entropy_mask])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_correct += int((y_hat[cross_entropy_mask].argmax(
                    dim=-1) == batch.y[cross_entropy_mask]).sum().cpu())

                total_examples += cross_entropy_mask.sum().cpu()
                total_loss += float(loss) * batch.batch_size

                pbar.set_postfix({"epoch": epoch, 'loss': total_loss / total_examples,
                                 'acc': total_correct / total_examples, 'tot': total_examples})
                pbar.update()
            scheduler.step()

        return total_loss / total_examples, total_correct / total_examples

    @torch.no_grad()
    def test(loader):
        encoder.eval()
        mlp_projection.eval()
        total_examples = total_correct = 0
        for batch in tqdm(loader):
            batch = batch.to(args.device)
            with torch.cuda.amp.autocast():
                batch = batch.to(args.device)
        #         h1, h2, h1_pred, h2_pred, h1_target, h2_target = encoder(batch, validation=True)
                h = encoder(batch, validation=True)

                # mlp_projection(torch.cat([h1, h2], dim=1))
                y_hat = mlp_projection(h)
                cross_entropy_mask = (batch.y >= 0)

        total_correct += int((y_hat[cross_entropy_mask].argmax(dim=-1)
                             == batch.y[cross_entropy_mask]).sum().cpu())
        total_examples += cross_entropy_mask.sum().cpu()

        return total_correct / total_examples

    def train_test(num_epochs):
        train_acc_res = []
        valid_acc_res = []
        losses = []

        for epoch in range(1, num_epochs+1):
            loss, train_acc = train(epoch)
            valid_acc = test(val_loader)

            train_acc_res.append(train_acc)
            valid_acc_res.append(valid_acc)
            losses.append(loss)
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

        return train_acc_res, valid_acc_res, losses

    train_test(15)

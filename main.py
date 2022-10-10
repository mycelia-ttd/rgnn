import numpy as np
import pandas as pd
from ogb.lsc import MAG240MDataset, MAG240MEvaluator

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

from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer, seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler



class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]
    year_id: Tensor

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
            year_id=self.year_id.to(*args, **kwargs)
        )

class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int], in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        return 128 + 64 + 64

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        # Trick A - remove irrelvalent training samples
        train_idx = dataset.get_idx_split('train')
        train_idx = train_idx[(dataset.all_paper_year[train_idx] > 2008) & (dataset.paper_label[train_idx].astype(int) != 100)]

        self.train_idx = torch.from_numpy(train_idx)
        print("train_idx size: ", self.train_idx.shape)
        del train_idx

        # Trick D - add label distribution embedding
        self.all_paper_year = np.concatenate((dataset.all_paper_year, np.full((dataset.num_authors + dataset.num_institutions, ), 2009))) #D

        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))

        self.train_idx.share_memory_()
        self.val_idx.share_memory_()
        self.test_idx.share_memory_()

        self.y = torch.from_numpy(dataset.all_paper_label).clone()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        if self.in_memory:
            print("in_memory")
            self.x = np.load('/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/full_feat_pai_m2v.npy')
            self.x = torch.from_numpy(self.x).share_memory_()
        else:
            self.x = x

        print("load adj_t")
        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=8)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=8)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=8)

    def convert_batch(self, batch_size, n_id, adjs):
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)

        year_id = torch.from_numpy(self.all_paper_year[n_id])
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs], year_id=year_id)

class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int, label_dist_emb:None,
                 heads: int = 4, dropout: float = 0.5, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'rgat':
            self.convs.append(
                ModuleList([
                    GATConv(in_channels, hidden_channels // heads, heads,
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

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

        # Trick D - Add label distribution embedding
        self.label_dist_emb = label_dist_emb
        self.label_dist_emb_mlp = Sequential(
            Linear(153, 64),
            BatchNorm1d(64),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
        )
        self.min_year = 2009

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor], year_id: Tensor, save_emb=False) -> Tensor:
        year_input = year_id - self.min_year
        year_input = torch.where(year_input < 0, 0, year_input)

        prev_year_ld_emb = self.label_dist_emb(year_input).float()
        prev_year_ld_emb = self.label_dist_emb_mlp(prev_year_ld_emb)

        x = torch.cat([x, prev_year_ld_emb], 1)

        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                subadj_t = subadj_t.set_value(None, layout=None)
                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)
                    # x = self.norms[i](out)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if save_emb:
            return x
        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.year_id)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.year_id)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.year_id)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]

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



ROOT = "/dbfs/mnt/ogb2022"
dataset = MAG240MDataset(root = ROOT)

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


class args:
    hidden_channels = 1024
    batch_size = 1024
    dropout = 0.5
    epochs = 100
    model = 'rgat'
    sizes = [25,15]
    in_memory = True
    device = 1
    evaluate = False


# ------------------------------------------ Train ----------------------------------------------
experiment_name = "rgat_submission"

seed_everything(42)
datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

experiment_name = "rgat_128_D"

logger_csv = CSVLogger("/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/logs/", name=experiment_name)
logger_tb_dbfs = TensorBoardLogger("/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/logs/tb/", name=experiment_name)
logger_tb_sys = TensorBoardLogger("logs", name=experiment_name)

model = RGNN(args.model, datamodule.num_features,
             datamodule.num_classes, args.hidden_channels,
             datamodule.num_relations, num_layers=len(args.sizes),
             label_dist_emb=label_dist_emb)
print(f'#Params {sum([p.numel() for p in model.parameters()])}')

ckp_path = f'{logger_csv}/ckpt/{experiment_name}'
checkpoint_callback = ModelCheckpoint(dirpath=ckp_path, monitor='val_acc', mode='max',save_top_k=1, save_last=True)

trainer = Trainer(accelerator='gpu',
                  devices=4,
                  strategy='horovod',
                  max_epochs=args.epochs,
                  logger=[logger_csv, logger_tb_dbfs, logger_tb_sys],
                  callbacks=[checkpoint_callback],
                  default_root_dir=f'logs/{args.model}')

trainer.fit(model, datamodule=datamodule)


# ------------------------------------------ Test ----------------------------------------------
dirs = glob.glob(f'/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/logs/{experiment_name}/*')

version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
logdir = f'/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/logs/{experiment_name}/version_{version}'

ckpt = glob.glob(f'{ckp_path}/*')[0]

print(f'Evaluating saved model in {ckp_path}...')
trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckp_path)
model = RGNN.load_from_checkpoint(checkpoint_path=ckpt,
                                 hparams_file=f'{logger_csv.log_dir}/hparams.yaml')

datamodule.batch_size = 16
datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

trainer.test(model=model, datamodule=datamodule)

evaluator = MAG240MEvaluator()
loader = datamodule.hidden_test_dataloader()

model.eval()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
model.to(device)
y_preds = []
for batch in tqdm(loader):
    batch = batch.to(device)
    with torch.no_grad():
        out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
        y_preds.append(out)
res = {'y_pred': torch.cat(y_preds, dim=0)}
evaluator.save_test_submission(res, "/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/logs/submission/", mode = 'test-challenge')

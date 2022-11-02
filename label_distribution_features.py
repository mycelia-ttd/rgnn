import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict

from ogb.lsc import MAG240MDataset
import numpy as np
import dgl
import torch


def calc_randomwalk_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        traces = np.unique(traces.numpy(), axis=0)  # deduplicate
        m = defaultdict(list)
        for sid, *_, did in traces:
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                lbs = labels[dids]
                mask = (dids != sid) & (lbs >= 0)
                if mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[mask, lbs[mask].astype(np.int64)] = 1
                    feat[i, :] = ft.sum(axis=0) / ft.sum()
            else:
                # TODO: Replace by year distribution
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_neighborsample_label_features(graph, node_ids, metapath, labels, num_classes=153):
    feat = np.zeros((len(node_ids), num_classes), dtype=np.float32)
    node_ids = np.array(node_ids)
    for i, nids0 in enumerate(tqdm.tqdm(node_ids)):
        nids = nids0
        for mp in metapath:
            _, nids = map(lambda x: x.numpy(), graph.out_edges(
                nids, form='uv', etype=mp))
        nids = np.unique(nids[(nids != nids0)])

        if len(nids) > 0:
            lbs = labels[nids]
            lbs = lbs[(lbs >= 0)].astype(np.int64)
            if len(lbs) == 0:
                feat[i, :] = 1. / num_classes
            else:
                ft = np.zeros((len(lbs), num_classes), dtype=np.float32)
                ft[list(range(len(lbs))), lbs] = 1
                feat[i, :] = ft.sum(axis=0) / ft.sum()
        else:
            feat[i, :] = 1. / num_classes
    return feat


def calc_randomwalk_topk_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, topk=10, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        traces = np.unique(traces.numpy(), axis=0)  # de-duplicate
        m = defaultdict(list)
        for sid, *_, did in traces:
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                lbs = labels[dids]
                mask = (dids != sid) & (lbs >= 0)
                if mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    dids, lbs = dids[mask], lbs[mask]
                    dids, indices, cnts = np.unique(
                        dids, return_index=True, return_counts=True)
                    lbs = lbs[indices]
                    itk = np.argsort(cnts)[-topk:]
                    dids, cnts, lbs = dids[itk], cnts[itk], lbs[itk]

                    mask = (dids != sid) & (lbs >= 0)
                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[mask, lbs[mask].astype(np.int64)] = 1
                    ft *= cnts.reshape((-1, 1))
                    feat[i, :] = ft.sum(axis=0) / cnts.sum()
            else:
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_neighborsample_filter_label_features_rw(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, topk=10, batch_size=2048, ftype='common', num_common=2):
    if ftype not in {'least', 'common', 'max'}:
        raise ValueError(
            "Unknown ftype: %r, only support 'least' and 'common'" % ftype)
    if len(metapath) != 2:
        raise ValueError("metapath should with length 2: %r" % metapath)

    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]
        traces = np.unique(traces.numpy(), axis=0)  # de-duplicate
        m = defaultdict(list)
        for sid, *_, did in traces:
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                lbs = labels[dids]
                mask = (dids != sid) & (lbs >= 0)
                if mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    dids, lbs = dids[mask], lbs[mask]
                    dids, indices, cnts = np.unique(
                        dids, return_index=True, return_counts=True)
                    lbs = lbs[indices]
                    if ftype == 'least':
                        itk = np.argsort(cnts)[:1]  # arg min
                    elif ftype == 'max':
                        itk = np.argsort(cnts)[-1:]  # arg max
                    else:
                        itk = cnts >= min(num_common, cnts.min())
                    dids, cnts, lbs = dids[itk], cnts[itk], lbs[itk]

                    mask = (dids != sid) & (lbs >= 0)
                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[mask, lbs[mask].astype(np.int64)] = 1
                    ft *= cnts.reshape((-1, 1))
                    feat[i, :] = ft.sum(axis=0) / cnts.sum()
            else:
                feat[i, :] = 1. / num_classes

        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_most_recent_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, topk=10, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]  # remove start == end
        traces = np.unique(traces.numpy(), axis=0)  # deduplicate
        m = defaultdict(list)
        for sid, *_, did in traces:
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                lbs = labels[dids]
                lb_mask = (dids != sid) & (lbs >= 0)
                dids, lbs = dids[lb_mask], lbs[lb_mask]
                if lb_mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    # top1, top 3, top 5?
                    itk = np.argsort(paper_year[dids])[-topk:]
                    dids = dids[itk]  # topk destinations by recency
                    lbs = lbs[itk]

                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[:, lbs.astype(np.int64)] = 1
                    feat[i, :] = ft.sum(axis=0) / ft.sum()
            else:
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


def calc_randomwalk_most_cited_label_features(graph, node_ids, metapath, labels, num_classes=153, num_walkers=160, topk=10, batch_size=1024):
    res = []
    for i in tqdm.tqdm(range(0, len(node_ids), batch_size)):
        nids0 = np.array(node_ids[i:(i + batch_size)])
        nids = np.repeat(nids0, num_walkers)
        traces, _ = dgl.sampling.random_walk(graph, nids, metapath=metapath)

        traces = traces[traces[:, 0] != traces[:, -1]]  # remove start == end
        traces = np.unique(traces.numpy(), axis=0)  # deduplicate
        m = defaultdict(list)
        for sid, *_, did in traces:
            if did >= 0:
                m[sid].append(did)
        mapper = {i: np.array(j) for i, j in m.items()}

        feat = np.zeros((len(nids0), num_classes), dtype=np.float32)
        for i, sid in enumerate(nids0):
            if sid in mapper.keys():
                dids = mapper[sid]
                lbs = labels[dids]
                lb_mask = (dids != sid) & (lbs >= 0)
                dids, lbs = dids[lb_mask], lbs[lb_mask]
                if lb_mask.sum() < 1.0e-6:
                    feat[i, :] = 1. / num_classes
                else:
                    # top1, top 3, top 5?
                    itk = np.argsort(number_of_citations[dids])[-topk:]
                    dids = dids[itk]  # topk destinations by recency
                    lbs = lbs[itk]

                    ft = np.zeros((len(dids), num_classes), dtype=np.float32)
                    ft[:, lbs.astype(np.int64)] = 1
                    feat[i, :] = ft.sum(axis=0) / ft.sum()
            else:
                feat[i, :] = 1. / num_classes
        res.append(feat)
    return np.concatenate(res, axis=0)


if __name__ == "__main__":
    ROOT = "/dbfs/mnt/ogb2022"
    dataset = MAG240MDataset(root=ROOT)

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test-whole')

    print('Building graph')
    # dataset = MAG240MDataset(root=args.rootdir)
    ei_writes = dataset.edge_index('author', 'writes', 'paper')
    ei_cites = dataset.edge_index('paper', 'paper')
    ei_affiliated = dataset.edge_index('author', 'institution')

    # different from dgl baseline, here has 6 edge types.
    g = dgl.heterograph({
        ('author', 'writes', 'paper'): (ei_writes[0], ei_writes[1]),
        ('paper', 'writed_by', 'author'): (ei_writes[1], ei_writes[0]),
        ('paper', 'cites', 'paper'): (ei_cites[0], ei_cites[1]),
        ('paper', 'cited_by', 'paper'): (ei_cites[1], ei_cites[0])
    })

    g = g.formats(['csr'])

    prefix = '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/submission'
    model_name = "labels_153"

    node_ids = np.concatenate(
        [dataset.get_idx_split(c).astype(np.int64)
         for c in ['train', 'valid', 'test-whole']]
    )

    num_classes = 153

    metapaths = {
        'pwbawp': ['writed_by', 'writes'],
        'pcpcbp': ['cites', 'cited_by'],
    }
    for n, mp in metapaths.items():
        print(n, mp)
        x = calc_neighborsample_filter_label_features_rw(g, node_ids, mp, paper_label, num_classes, num_walkers=420,
                                                         ftype='common', num_common=2)
        np.save(
            f'{prefix}/results/{model_name}/x_{n}_rw_c2_lratio_valid_full_v2.npy', x)

        x = calc_neighborsample_filter_label_features_rw(g, node_ids, mp, paper_label, num_classes, num_walkers=420,
                                                         ftype='least')
        np.save(
            f'{prefix}/results/{model_name}/x_{n}_rw_l_lratio_valid_full_v2.npy', x)

        x = calc_neighborsample_filter_label_features_rw(g, node_ids, mp, paper_label, num_classes, num_walkers=420,
                                                         ftype='max')
        np.save(
            f'{prefix}/results/{model_name}/x_{n}_rw_m_lratio_valid_full_v2.npy', x)

        x = calc_neighborsample_filter_label_features_rw(g, node_ids, mp, paper_label, num_classes, num_walkers=420,
                                                         ftype='common', num_common=3)
        np.save(
            f'{prefix}/results/{model_name}/x_{n}_rw_c3_lratio_valid_full_v2.npy', x)

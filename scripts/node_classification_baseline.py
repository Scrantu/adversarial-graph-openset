#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cora_node_classification.py: Static graph node classification on Cora using DGIBNN architecture.
"""
from torch_geometric.nn import GCNConv, GATConv
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import (
    add_remaining_self_loops,
    remove_self_loops,
    add_self_loops,
    softmax,
    degree,
    to_undirected,
)
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from DGIB.model_sythetic import DGIBNN
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
from torch_geometric.data import Data
import random
import nni

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import pickle

def make_openset_split_auto(data, known_class_ratio=0.7, train_ratio=0.1, val_ratio=0.1, seed=seed, save_path=None):
    """
    将部分类别作为 unknown 类，划分 openset 节点分类任务，并返回相关掩码。

    参数:
        data: PyG Data 对象，包含 data.y
        known_class_ratio: float, 用于训练的 known 类比例（如 0.7 表示前 70% 类为 known）
        train_ratio: float, 训练集中使用多少比例的 known-class 节点
        val_ratio: float, 验证集中使用多少比例（从非训练的 known 节点中再划分）

    返回:
        data: 更新后的 data 对象，包含重编码标签 + train/val/test_mask
    """

    labels = data.y.cpu().numpy()
    all_classes = np.unique(labels)
    num_classes = len(all_classes)

    num_known = int(num_classes * known_class_ratio)
    shuffled_classes = np.random.permutation(all_classes)
    known_classes = shuffled_classes[:num_known]
    unknown_classes = shuffled_classes[num_known:]

    print(f"Known classes: {known_classes.tolist()}")
    print(f"Unknown classes: {unknown_classes.tolist()}")

    # 构建 label remap: known class -> [0, ..., num_known-1], unknown -> num_known (统一unknown标签)
    label_map = {}
    for i, cls in enumerate(known_classes):
        label_map[cls] = i
    unknown_class_id = num_known
    for cls in unknown_classes:
        label_map[cls] = unknown_class_id
    print('label_map', label_map)

    new_labels = np.array([label_map[y] for y in labels])
    data.y = torch.from_numpy(new_labels).to(torch.long)

    known_mask = np.isin(new_labels, list(range(num_known)))
    unknown_mask = new_labels == unknown_class_id

    known_indices = np.where(known_mask)[0]
    unknown_indices = np.where(unknown_mask)[0]

    # 划分 train/val/test: train/val 来自 known, test 包含全部
    train_idx, rest_idx = train_test_split(
        known_indices, train_size=train_ratio, random_state=seed, stratify=new_labels[known_indices]
    )
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_known_idx = train_test_split(
        rest_idx, test_size=1 - val_size, random_state=seed, stratify=new_labels[rest_idx]
    )
    test_idx = np.concatenate([test_known_idx, unknown_indices])

    # 设置布尔掩码
    num_nodes = data.y.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 构造 test 内部的 known/unknown 掩码
    known_in_unseen_mask = torch.zeros(num_nodes, dtype=torch.bool)
    unknown_in_unseen_mask = torch.zeros(num_nodes, dtype=torch.bool)
    known_in_unseen_mask[test_known_idx] = True
    unknown_in_unseen_mask[unknown_indices] = True
    data.known_in_unseen_mask = known_in_unseen_mask
    data.unknown_in_unseen_mask = unknown_in_unseen_mask

    split_info = {
        'label_map': label_map,  # 可以保留 int 类型
        'idx_train': train_idx,
        'idx_val': val_idx,
        'idx_test': test_idx,
        'idx_known_unseen': test_known_idx,
        'idx_unknown_unseen': unknown_indices,
    }
    # 保存到文件
    with open(f'{save_path}', 'wb') as f:
        pickle.dump(split_info, f)
    # with open('openset_split.pkl', 'rb') as f:
    #     split_info = pickle.load(f)

    return data, num_known, num_classes-num_known


def load_npz_as_pyg_data(file_name, is_sparse=True, require_lcc=True):
    with np.load(file_name) as loader:
        adj = sp.csr_matrix(
                (loader['data'], loader['indices'], loader['indptr']),
                shape=loader['shape']
            )
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    return edge_index

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level= 0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, threshould

def train(model, data, optimizer, device, args):
    model.train()
    optimizer.zero_grad()
    x_all = data.x.to(device)
    edge_index_all = data.edge_index.to(device)
    # Forward through DGIBNN
    embeddings = model(x_all, edge_index_all)

    outs = F.log_softmax(embeddings, dim=1)
    # Compute cross-entropy loss on train mask
    y_true = data.y.to(device)
    loss_cls = F.nll_loss(outs[data.train_mask], y_true[data.train_mask])
    # Combine with DGIB losses
    loss = loss_cls 

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, device):
    model.eval()
    x_all = data.x.to(device)
    edge_index_all = data.edge_index.to(device)
    logits = model(x_all, edge_index_all)
    outs = F.log_softmax(logits, dim=1)
    # print('outs', outs)
    y_true = data.y.to(device)
    accs = {}
    for split in ['train', 'val', 'test', 'known_in_unseen']:
        mask = getattr(data, f"{split}_mask").to(device)
        accs[split] = accuracy(outs[mask], y_true[mask])
    test_ind_score, test_openset_score = model.detect(logits.to(device), data.to(device))
    auroc, aupr, fpr, _ = get_measures(test_ind_score.cpu(), test_openset_score.cpu())
    accs['openset'] = [auroc] + [aupr] + [fpr]

    return accs

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def detect(self, logits, data, T=1.0):
        neg_energy = T * torch.logsumexp(logits / T, dim=-1)
        neg_energy = self.propagation(neg_energy, data.edge_index)
        ind_idx, openset_idx = data.known_in_unseen_mask, data.unknown_in_unseen_mask

        neg_energy_ind = neg_energy[ind_idx]
        neg_energy_openset = neg_energy[openset_idx]
        return neg_energy_ind, neg_energy_openset
    
    def propagation(self, e, edge_index, prop_layers=2, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)


def main():
    parser = argparse.ArgumentParser(description='DGIBNN on static Cora for node classification')
    # Model hyperparameters
    parser.add_argument('--n_layers', type=int, default=2, help='Number of DGIBConv layers')
    parser.add_argument('--nhid', type=int, default=128, help='Hidden (latent) size')
    parser.add_argument('--nout', type=int, default=-1, help='Output (class) size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    parser.add_argument('--reparam_mode', type=str, default='diag', help='Reparameterization mode for XIB. Choose from "None", "diag" or "full"')
    parser.add_argument('--prior_mode', type=str, default='Gaussian', help='Prior mode. Choose from "Gaussian" or "mixGau-10" (mixture of 10 Gaussian components)')
    parser.add_argument('--distribution', type=str, default='Bernoulli', help="categorical,Bernoulli")
    parser.add_argument('--temperature', type=float, default=1, help='Sampling temperature')
    parser.add_argument('--nbsz', type=int, default=20, help='Neighbor sample size')
    parser.add_argument('--sample_size', type=int, default=50, help='Reparameterize samples')
    # Loss weights
    # good:(0.01, 0.001)
    parser.add_argument('--lambda_ixz', type=float, default=0.00, help='Weight for I(X;Z) loss')
    parser.add_argument('--lambda_struct', type=float, default=0.00, help='Weight for structure KL loss')
    parser.add_argument('--lambda_cons', type=float, default=0.0, help='Weight for consensual loss')
    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Saved model')
    parser.add_argument('--dataset', type=str, default='citeseer', help='data name')

    args = parser.parse_args()

    # Load Cora dataset
    dataset = Planetoid(root='../data/cora', name=args.dataset)
    data = dataset[0]

    known_class_ratio=0.7
    train_ratio=0.1
    val_ratio=0.1
    data, num_known_classes, num_unknown_classes = make_openset_split_auto(
        data, known_class_ratio=known_class_ratio, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed, 
        save_path=f"{args.dataset}-{str(known_class_ratio)}-{str(train_ratio)}-{str(val_ratio)}-{str(seed)}.pkl")
    print('num_known_classes', num_known_classes, num_unknown_classes)
    adj = load_npz_as_pyg_data(f"C:/Users/zhaoc/Desktop/DeepRobust-latest/examples/graph/data/mod_graph_{args.dataset}-0.7-0.1-0.1-0-MetaSelf-0.15-gnnsafe/mod_adj.npz")

     # 对比
    ###################
    from torch_geometric.utils import to_undirected
    def edge_index_to_set(edge_index):
        """Convert edge_index to a set of (min, max) undirected edge tuples"""
        edge_index = to_undirected(edge_index)
        edge_index = edge_index.cpu().numpy()
        edge_tuples = set()
        for src, dst in zip(edge_index[0], edge_index[1]):
            # 把无向边按 (min, max) 存储，避免重复
            edge_tuples.add((min(src, dst), max(src, dst)))
        return edge_tuples

    # 假设 data.edge_index 是原始的，adj 是加载出来的新 edge_index
    original_edge_set = edge_index_to_set(data.edge_index)
    mod_edge_set = edge_index_to_set(adj)

    edges_only_in_original = original_edge_set - mod_edge_set
    edges_only_in_mod = mod_edge_set - original_edge_set
    intersection = original_edge_set & mod_edge_set

    print(f"Original edge count: {len(original_edge_set)}")
    print(f"Modified edge count: {len(mod_edge_set)}")
    print(f"Common edges: {len(intersection)}")
    print(f"Edges only in original: {len(edges_only_in_original)}")
    print(f"Edges only in modified: {len(edges_only_in_mod)}")

    # 节点索引集合
    train_set = set(data.train_mask.nonzero(as_tuple=True)[0].tolist())
    test_set = set(data.test_mask.nonzero(as_tuple=True)[0].tolist())

    # 统计函数
    def count_edge_types(edge_set, train_set, test_set):
        tt, tr_te, te_te = 0, 0, 0
        for u, v in edge_set:
            if u in train_set and v in train_set:
                tt += 1
            elif (u in train_set and v in test_set) or (v in train_set and u in test_set):
                tr_te += 1
            elif u in test_set and v in test_set:
                te_te += 1
        total = len(edge_set)
        return {
            'train-train': tt / total if total > 0 else 0,
            'train-test': tr_te / total if total > 0 else 0,
            'test-test': te_te / total if total > 0 else 0,
            'total': total
        }

    # 分别统计新增边和删除边的比例分布
    added_stats = count_edge_types(edges_only_in_mod, train_set, test_set)
    removed_stats = count_edge_types(edges_only_in_original, train_set, test_set)

    # 打印结果
    print("== Modified Edge Stats ==")
    print(f"Original edge count: {len(original_edge_set)}")
    print(f"Modified edge count: {len(mod_edge_set)}")
    print(f"Common edges: {len(original_edge_set & mod_edge_set)}")
    print(f"Edges only in original (deleted): {len(edges_only_in_original)}")
    print(f"Edges only in modified (added): {len(edges_only_in_mod)}\n")

    print("== Added Edge Distribution ==")
    for k, v in added_stats.items():
        print(f"{k}: {v:.2%}")

    print("\n== Removed Edge Distribution ==")
    for k, v in removed_stats.items():
        print(f"{k}: {v:.2%}")
    ########################


    data.edge_index = adj

    device = torch.device(args.device)
    # Initialize DGIBNN model
    args.nfeat = dataset.num_features
    args.nhid = args.nhid
    args.n_layers = args.n_layers
    args.nout = num_known_classes
    model = GCN(dataset.num_features, args.nhid, num_known_classes, dropout=args.dropout).to(device)


    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    patience_counter = 0
    best_val = 0.0
    best_overall_test = 0.0
    best_known_in_test = 0.0
    best_openset_metrics = None
    id_test_acc_list = []       # known_in_unseen 精度
    id_val_acc_list = []
    ood_auroc_list = [] 
    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1)):
        loss = train(model, data, optimizer, device, args)
        accs = test(model, data, device)
        if accs['val'] > best_val:
            best_val = accs['val']
            best_overall_test = accs['test']
            best_known_in_test = accs['known_in_unseen']
            best_openset_metrics = accs['openset']
            # torch.save({
            #             'epoch': epoch,
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'val_acc': best_val
            #             }, args.save_path)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {accs['train']:.4f} "  
                f"| Val: {accs['val']:.4f} | Overall Test: {accs['test']:.4f} | Known in Test: {accs['known_in_unseen']:.4f} | Openset detection: {accs['openset']}")
            # model.visualize_energy_distributions(neg_energy_ind, neg_energy_openset, title="Energy Distribution")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
        
        id_test_acc_list.append(accs['known_in_unseen'].cpu())
        id_val_acc_list.append(accs['val'].cpu())
        ood_auroc_list.append(accs['openset'][0])  # AUROC


    # Load best model and evaluate
    print(f"\nBest validation/test overall accuracy/test ind accuracy/openset detection: {best_val:.4f}/{best_overall_test:.4f}/{best_known_in_test:.4f}/{best_openset_metrics}, model saved to {args.save_path}")
    
    # checkpoint = torch.load(args.save_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # accs = test(model, data, device)
    # print("\n=== Test after loading best model ===")
    # print(f"Train: {accs['train']:.4f} | Val: {accs['val']:.4f} | Test: {accs['test']:.4f}")


    plt.figure()
    plt.plot(id_test_acc_list, label="ID Accuracy (Known in Test)")
    plt.plot(id_val_acc_list, label="ID Accuracy (Val)")
    plt.plot(ood_auroc_list, label="OOD Detection AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.legend()
    plt.title("ID Classification vs OOD Detection")
    plt.grid(True)
    plt.savefig("id_vs_ood_performance.png")
    plt.show()

if __name__ == '__main__':
    main()

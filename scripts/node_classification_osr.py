#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cora_node_classification.py: Static graph node classification on Cora using DGIBNN architecture.
"""

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
from DGIB.model_osr import *
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
    # print(f"Train nodes: {(train_idx)}, Val nodes: {(val_idx)}, Test nodes: {(test_idx)}")
    # input("Press Enter to continue...")
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

energy_gap_list = []

ixz = []
ia = []
i_both = []
i_yz_list = []
i_yoodz_list = []

ixz_init = None
ia_init = None
i_both_init = None
iyz_init = None
iyoodz_init = None


def train(model: DGIBNN_mlt, data, optimizer, device, args, epoch):
    model.train()
    optimizer.zero_grad()
    x_all = [data.x.to(device)]
    edge_index_all = [data.edge_index.to(device)]
    # Forward through DGIBNN
    embed_last_hid, logits, ixz_loss_s, struct_kl_loss_s = model(x_all, edge_index_all)
    logits = logits[0]
    embed_last_hid = embed_last_hid[0]

    outs = F.log_softmax(logits, dim=1)
    # Compute cross-entropy loss on train mask
    y_true = data.y.to(device)
    loss_cls = F.nll_loss(outs[data.train_mask], y_true[data.train_mask])

    # pu loss
    mu = 0.5
    loss_pu = model.pu_discriminator_loss(embed_last_hid[data.train_mask], embed_last_hid[data.test_mask], mu = mu)
    
    # Combine with DGIB losses
    lambda_pu = min(1.0, epoch / 50) * 0.1
    loss = loss_cls \
           + args.lambda_ixz * ixz_loss_s \
           + args.lambda_struct * struct_kl_loss_s \
           + lambda_pu * loss_pu


    # openset loss
    with torch.no_grad():
        probs_all = model.d_phi(embed_last_hid).detach()

        weights_all = model.compute_openset_weight(probs_all).detach()
        mask_cand_ood = model.select_topk_ood_candidates(weights_all, mu, data.test_mask).detach()

        # mask_cand_id = model.select_topk_id_candidates(weights_all, 1-mu, data.test_mask).detach()
        # mask_cand_ood2 = model.select_topk_id_candidates(weights_all, 1-mu, data.test_mask).detach()

    if epoch > 0:
        # --------------------------- 伪 OOD 对齐（加权版本） ---------------------------
        # model.d_phi.eval()
        z_pos = embed_last_hid[data.train_mask]  # 假设第一个分支输出是 ID 嵌入
        logits_pos = logits[data.train_mask]
        energy_ind = - torch.logsumexp(logits_pos, dim=-1).squeeze()
        # print('energy_ind', energy_ind)


        # # -----------------初始化伪ood--------------------
        # train_probs = F.softmax(logits_pos, dim=1)
        # train_labels = y_true[data.train_mask]
        # train_embeds = embed_last_hid[data.train_mask]
        
        # # 计算每个类别需要的样本数
        # num_classes = args.nout
        # samples_per_class = max(args.sample_size // num_classes + 1, 5)  # 每类至少取5个样本
        
        # # 对每个类别找到预测置信度最低的多个节点
        # x_init_list = []
        # for class_idx in range(num_classes):
        #     class_mask = (train_labels == class_idx)
        #     if class_mask.sum() > 0:
        #         class_probs = train_probs[class_mask]
        #         class_embeds = train_embeds[class_mask]
        #         # 获取每个样本对其真实类别的预测置信度
        #         class_conf = class_probs[:, class_idx]
        #         # 选择置信度最低的k个样本的嵌入
        #         k = min(samples_per_class, len(class_conf))  # 防止样本数不足
        #         low_conf_idx = torch.argsort(class_conf)[:k]
        #         x_init_list.extend(class_embeds[low_conf_idx])
        
        # # 将列表转换为张量
        # x_init = torch.stack(x_init_list)  # [n_classes * samples_per_class, hidden_dim]
        
        # # 如果收集的样本数超过sample_size，随机采样
        # if len(x_init) > args.sample_size:
        #     perm = torch.randperm(len(x_init))
        #     x_init = x_init[perm[:args.sample_size]]
        # # 如果样本数不足，通过重复来补足
        # elif len(x_init) < args.sample_size:
        #     repeat_times = args.sample_size // len(x_init) + 1
        #     x_init = x_init.repeat(repeat_times, 1)[:args.sample_size]


        z_cand_ood = embed_last_hid[mask_cand_ood]
        weights_cand = weights_all[mask_cand_ood].detach()
        z_sample_ood = model.sample(
            sample_size=len(z_pos),
            max_buffer_len=int(model.max_buffer_vol * len(z_pos)),
            device=args.device,
            x_cand=z_cand_ood,  # 加入候选伪OOD
            weights_cand=weights_cand,
            lambda_align=0.2,
            x_init=None,
        )
        z_sample_ood = F.elu(z_sample_ood)
        z_sample_ood = F.normalize(z_sample_ood, dim=-1)
        logits_pop = model.branch1_cls(z_sample_ood)
        energy_pop = - torch.logsumexp(logits_pop, dim=-1).squeeze()

        # z_cand_id = embed_last_hid[mask_cand_id]
        # weights_cand = weights_all[mask_cand_id].detach()
        # z_sample_id = model.sample(
        #     sample_size=len(z_pos),
        #     max_buffer_len=int(model.max_buffer_vol * len(z_pos)),
        #     device=args.device,
        #     x_cand=z_cand_id,  # 加入候选伪ID
        #     weights_cand=weights_cand,
        #     lambda_align=0.2
        # )
        # z_sample_id = F.elu(z_sample_id)
        # z_sample_id = F.normalize(z_sample_id, dim=-1)
        # logits_pid = model.branch1_cls(z_sample_id)
        # energy_pid = - torch.logsumexp(logits_pid, dim=-1).squeeze()
        # if epoch % 10 == 0:
        #     visualize_embeddings_tsne(z_all, z_sample, data)

        loss_uncertainty = torch.mean(F.relu(energy_ind - 0) ** 2 + F.relu(1 - energy_pop) ** 2) \
                        #  + torch.mean(F.relu(energy_pid - 0) ** 2 + F.relu(1 - energy_pop) ** 2) \
                        #  + torch.mean(F.relu(energy_ind - energy_pop) ** 2) \
        
        # loss_reg = torch.mean(energy_ind.pow(2) + energy_pop.pow(2) + energy_pid.pow(2))
        loss_reg = torch.mean(energy_ind.pow(2) + energy_pop.pow(2))
        # loss_uncertainty = model.loss_uncertainty_softmargin(energy_ind, energy_pop)
        # loss_uncertainty = (energy_ind - energy_sample).mean()
        lambda_uncertainty = min(1.0, (epoch - 20) / 50) * 0.1 if epoch > 20 else 0.0
        lambda_reg = min(1.0, (epoch - 20) / 50) * 0.01 if epoch > 20 else 0.0
        loss = loss + lambda_uncertainty * loss_uncertainty + lambda_reg * loss_reg
    
    global energy_gap_list
    energy_gap_list.append(-(outs[data.train_mask] * torch.exp(outs[data.train_mask])).sum(dim=1).mean().item())
    
    loss.backward()
    optimizer.step()


    # ====== closeset: I(Y;Z) 计算 ======
    probs = torch.softmax(logits, dim=1)  # [N, num_classes]
    mask = data.train_mask
    probs_masked = probs[mask]
    y_masked = y_true[mask]
    num_classes = probs.size(1)
    labels_onehot = F.one_hot(y_masked, num_classes=num_classes).float()
    label_dist = labels_onehot.mean(dim=0)  # 类别分布
    H_Y = -(label_dist * torch.log(label_dist + 1e-12)).sum().item() / np.log(2)  # bit
    H_Y_given_Z = -(probs_masked * torch.log(probs_masked + 1e-12)).sum(dim=1).mean().item() / np.log(2)  # bit
    i_yz = H_Y - H_Y_given_Z

    # ====== openset: I(Y;Z) 计算 ======
    mask_test = data.test_mask
    id_mask_test = data.known_in_unseen_mask & data.test_mask
    ood_mask_test = data.unknown_in_unseen_mask & data.test_mask

    # 能量分数
    energy_test_all = torch.logsumexp(logits[mask_test], dim=-1)
    energy_id = torch.logsumexp(logits[id_mask_test], dim=-1)

    # 阈值（FPR95）
    threshold = np.percentile(energy_id.detach().cpu().numpy(), 5)
    alpha_energy = 10.0

    # OOD 概率
    p_ood_score = alpha_energy * (energy_test_all - threshold)
    probs_ood = torch.stack([1 - p_ood_score, p_ood_score], dim=-1)

    # OOD 标签（映射到测试集内）
    y_ood_test = torch.zeros(mask_test.sum(), dtype=torch.long, device=device)
    y_ood_test[ood_mask_test[mask_test]] = 1

    # I(Y_ood; Z)
    labels_onehot_ood = F.one_hot(y_ood_test, num_classes=2).float()
    label_dist_ood = labels_onehot_ood.mean(dim=0)
    H_Yood = -(label_dist_ood * torch.log(label_dist_ood + 1e-12)).sum().item() / np.log(2)
    H_Yood_given_Z = -(probs_ood * torch.log(probs_ood + 1e-12)).sum(dim=1).mean().item() / np.log(2)
    i_yoodz = H_Yood - H_Yood_given_Z

    # ================= 初始化平移基准 =================
    global ixz_init, ia_init, i_both_init, iyz_init, iyoodz_init
    if ixz_init is None:
        ixz_init = ixz_loss_s.item()
        ia_init = struct_kl_loss_s.item()
        i_both_init = ixz_init + ia_init
        iyz_init = i_yz
        iyoodz_init = i_yoodz

    # ================= 存值（bit单位 + 平移） =================
    ixz.append((ixz_loss_s.item() - ixz_init) / np.log(2))
    ia.append((struct_kl_loss_s.item() - ia_init) / np.log(2))
    i_both.append((ixz_loss_s.item() + struct_kl_loss_s.item() - i_both_init) / np.log(2))
    i_yz_list.append(i_yz - iyz_init)
    i_yoodz_list.append(i_yoodz - iyoodz_init)


    return loss.item()

# @torch.no_grad()
# def test(model: DGIBNN_mlt, data, device, epoch):
#     model.eval()
#     x_all = [data.x.to(device)]
#     edge_index_all = [data.edge_index.to(device)]
#     embed_last_hid, logits, _, _ = model(x_all, edge_index_all)
#     logits = logits[0]
#     outs = F.log_softmax(logits, dim=1)
#     y_true = data.y.to(device)
#     accs = {}
    
#     # 常规分类准确率评估
#     for split in ['train', 'val', 'test']:
#         mask = getattr(data, f"{split}_mask").to(device)
#         accs[split] = accuracy(outs[mask], y_true[mask])

#     # Openset detection评估
#     test_ind_score, test_openset_score = model.detect(logits.to(device), data.to(device))
#     auroc, aupr, fpr, threshold = get_measures(test_ind_score.cpu(), test_openset_score.cpu())
#     accs['openset'] = [auroc, aupr, fpr]

#     if epoch % 10 == 0:
#         visualize_energy_distributions(test_ind_score, test_openset_score, title="Energy Distribution")
#         print('threshold', threshold)

#     # 修改这部分逻辑
#     test_mask = data.test_mask
#     test_scores = torch.cat([test_ind_score, test_openset_score])
    
#     # 1. 根据score判定ID/OOD
#     is_id_pred = (test_scores > threshold)  # True表示是ID样本
#     # 2. 对已知类 + OOD类进行多分类预测
#     final_preds = outs[test_mask].argmax(dim=1)
    
#     # 3. 只有当预测为ID样本时才使用分类预测结果
#     # 如果预测为OOD，则标记为unknown_class
#     unknown_class_id = data.y.max().item()
#     final_preds[~is_id_pred] = unknown_class_id
    
#     # 获取真实标签
#     y_true_test = data.y[test_mask]
    
#     # 计算多分类的准确率和F1分数
#     multi_acc = (final_preds == y_true_test).float().mean()
#     f1_macro = f1_score(y_true_test.cpu(), final_preds.cpu(), average='macro')
#     f1_weighted = f1_score(y_true_test.cpu(), final_preds.cpu(), average='weighted')
    
#     accs['multi_class'] = {
#         'acc': multi_acc.item(),
#         'f1_macro': f1_macro,
#         'f1_weighted': f1_weighted
#     }

#     # Pseudo openset评估(如果需要)
#     mu = 0.5
#     with torch.no_grad():
#         probs_all = torch.sigmoid(model.d_phi(embed_last_hid[0]).detach())
#         weights_all = model.compute_openset_weight(probs_all).detach()
#         mask_cand = model.select_topk_ood_candidates(weights_all, 0.7, data.test_mask).detach()
#     e = model.get_energy_score(logits.to(device), data.to(device))
#     test_pseudo_openset_score = e[mask_cand]
#     auroc, aupr, fpr, _ = get_measures(test_ind_score.cpu(), test_pseudo_openset_score.cpu())
#     accs['pseudo_openset'] = [auroc, aupr, fpr]

#     return accs

@torch.no_grad()
def test(model: DGIBNN_mlt, data, device, epoch):
    model.eval()
    x_all = [data.x.to(device)]
    edge_index_all = [data.edge_index.to(device)]
    embed_last_hid, logits, _, _ = model(x_all, edge_index_all)
    logits = logits[0]
    probs = F.softmax(logits, dim=1)
    outs = probs.log()
    y_true = data.y.to(device)
    accs = {}
    
    # 常规分类准确率评估
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f"{split}_mask").to(device)
        accs[split] = accuracy(outs[mask], y_true[mask])


    # 从测试集中选择OOD分数最小的部分样本作为伪OOD
    ood_scores = model.detect(logits.to(device), data.to(device))
    test_scores = ood_scores[data.test_mask]
    pseudo_ratio = 0.2  # 选择分数最小的20%作为伪OOD
    pseudo_threshold = torch.quantile(test_scores, pseudo_ratio)
    pseudo_ood_mask = test_scores < pseudo_threshold

    # print("Length of pseudo_ood_mask:", len(pseudo_ood_mask), pseudo_ood_mask)
    # print('length of val mask', len(data.val_mask), data.val_mask)


    # 获取验证集和伪OOD样本
    val_mask = data.val_mask
    y_true = data.y.to(device)

    # 合并真实ID样本和伪OOD样本
    val_true_labels = torch.ones_like(ood_scores[val_mask]).float()  # 所有验证集样本的真实标签都是1（ID）
    pseudo_ood_true_labels = torch.zeros_like(test_scores[pseudo_ood_mask]).float()  # 伪OOD标签为0

    # 合并标签和分数
    true_labels = torch.cat([val_true_labels, pseudo_ood_true_labels])  # 确保合并的标签数量一致
    scores = torch.cat([ood_scores[val_mask], test_scores[pseudo_ood_mask]])  # 确保合并的分数数量一致
    # print("Length of true_labels:", len(true_labels))
    # print("Length of scores:", len(scores))



    # 在这里检查标签和分数的长度
    if len(true_labels) != len(scores):
        raise ValueError(f"Length of true_labels ({len(true_labels)}) and scores ({len(scores)}) are not equal!")

    # 计算AUROC
    accs['overall_val'] = 0.7 * roc_auc_score(true_labels.cpu(), scores.cpu()) + 0.3 * accs['val']





    # 计算所有节点的OOD分数
    ood_scores = model.detect(logits.to(device), data.to(device))

    # 使用训练集的分数来确定阈值
    train_scores = ood_scores[data.train_mask]
    
    # 为了评估AUROC，我们需要测试集的真实标签
    test_scores = ood_scores[data.test_mask]
    test_true_labels = (y_true[data.test_mask] != data.y.max().item()).cpu()  # 1表示OOD（移除了~）
    auroc = roc_auc_score(test_true_labels, test_scores.cpu())  

    # 用训练集分数计算阈值（5%分位数，因为我们假设训练集都是ID样本）
    threshold = torch.quantile(train_scores, 0.05)  # 改为5%分位数


    accs['openset'] = [auroc, 0, 0]
    
    if epoch % 10 == 0:
        print('threshold from train scores:', threshold.item())

    # 使用训练集确定的阈值进行OOD检测
    test_mask = data.test_mask
    is_ood_pred = (ood_scores[test_mask] < threshold).long()  # 改为小于号


    # 多分类预测
    final_preds = probs[test_mask].argmax(dim=1)
    unknown_class_id = data.y.max().item()
    final_preds[is_ood_pred == 1] = unknown_class_id  # OOD样本标记为unknown_class
    
    # 获取测试集的真实标签
    y_true_test = y_true[test_mask]
    
    # 计算多分类的准确率和F1分数
    multi_acc = (final_preds == y_true_test).float().mean()
    f1_macro = f1_score(y_true_test.cpu(), final_preds.cpu(), average='macro')
    f1_weighted = f1_score(y_true_test.cpu(), final_preds.cpu(), average='weighted')
    
    # 分别计算已知类和OOD类的指标
    known_mask = y_true_test != unknown_class_id
    ood_mask = y_true_test == unknown_class_id

     # 已知类指标
    if known_mask.sum() > 0:
        known_acc = (final_preds[known_mask] == y_true_test[known_mask]).float().mean()
        known_f1 = f1_score(y_true_test[known_mask].cpu(), final_preds[known_mask].cpu(), average='macro')
    else:
        known_acc = known_f1 = 0.0
        
    # OOD类指标
    if ood_mask.sum() > 0:
        ood_acc = (final_preds[ood_mask] == y_true_test[ood_mask]).float().mean()
        # OOD只有一个类，直接用二分类F1
        ood_f1 = f1_score(
            (y_true_test[ood_mask] == unknown_class_id).cpu(),
            (final_preds[ood_mask] == unknown_class_id).cpu(),
            average='binary'
        )
    else:
        ood_acc = ood_f1 = 0.0
    
    accs['multi_class'] = {
        'acc': multi_acc.item(),
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'known_acc': known_acc.item() if torch.is_tensor(known_acc) else known_acc,
        'known_f1': known_f1,
        'ood_acc': ood_acc.item() if torch.is_tensor(ood_acc) else ood_acc,
        'ood_f1': ood_f1
    }
    accs['pseudo_openset'] = [0, 0, 0]  # Pseudo openset评估可以根据需要添加
    return accs

def debug_structure_kl(model, epoch):
    """
    调试 DGIBConv 中的 IA Loss（structure_kl_loss）
    """
    print(f"\n[Epoch {epoch}] --- Structure KL Loss 调试 ---")

    for name, module in model.named_modules():
        if isinstance(module, DGIBConv):  # 只处理 DGIBConv 层
            # structure_kl_loss
            if hasattr(module, "structure_kl_loss_list"):
                kl_vals = module.structure_kl_loss_list
                if isinstance(kl_vals, list) and len(kl_vals) > 0:
                    kl_vals = [float(v) if torch.is_tensor(v) else float(v) for v in kl_vals]
                    print(f"[{name}] structure_kl_loss_list: mean={np.mean(kl_vals):.6f}, "
                          f"std={np.std(kl_vals):.6f}, min={np.min(kl_vals):.6f}, max={np.max(kl_vals):.6f}")

            # alpha 分布
            if hasattr(module, "alpha"):
                alpha_np = module.alpha.detach().cpu().numpy().flatten()
                print(f"[{name}] alpha: mean={alpha_np.mean():.6f}, std={alpha_np.std():.6f}, "
                      f"min={alpha_np.min():.6f}, max={alpha_np.max():.6f}")

                # 可视化分布
                plt.hist(alpha_np, bins=50)
                plt.title(f"Alpha distribution - {name} (Epoch {epoch})")
                plt.xlabel("alpha value")
                plt.ylabel("count")
                plt.show()

            # attention 参数梯度
            if hasattr(module, "att"):
                if module.att.grad is not None:
                    grad_np = module.att.grad.detach().cpu().numpy().flatten()
                    print(f"[{name}] att.grad: mean={grad_np.mean():.6f}, std={grad_np.std():.6f}, "
                          f"min={grad_np.min():.6f}, max={grad_np.max():.6f}")
                else:
                    print(f"[{name}] att.grad = None (梯度没传到注意力参数)")


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
    parser.add_argument('--distribution', type=str, default='categorical', help="categorical,Bernoulli")
    parser.add_argument('--temperature', type=float, default=1, help='Sampling temperature')
    parser.add_argument('--nbsz', type=int, default=20, help='Neighbor sample size')
    parser.add_argument('--sample_size', type=int, default=50, help='Reparameterize samples')
    # Loss weights
    # good:(0.01, 0.001)
    parser.add_argument('--lambda_ixz', type=float, default=0.005, help='Weight for I(X;Z) loss')
    parser.add_argument('--lambda_struct', type=float, default=0.1, help='Weight for structure KL loss')
    parser.add_argument('--lambda_cons', type=float, default=0.0, help='Weight for consensual loss')
    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=250, help='Early stopping patience')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Saved model')
    parser.add_argument('--dataset', type=str, default='citeseer', help='data name')
    parser.add_argument('--attack', type=str, default='MetaSelf', help='dice, MetaSelf')

    args = parser.parse_args()

    if args.dataset == 'citeseer':
        args.distribution = 'Bernoulli'
    elif args.dataset == 'cora':
        args.distribution = 'categorical'

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
    if args.attack == 'MetaSelf':
        adj = load_npz_as_pyg_data(f"C:/Users/zhaoc/Desktop/DeepRobust-latest/examples/graph/data/mod_graph_{args.dataset}-0.7-0.1-0.1-0-MetaSelf-0.15-gnnsafe/mod_adj.npz")
    elif args.attack == 'dice':
        adj = torch.load(f"C:/Users/zhaoc/Desktop/DeepRobust-latest/examples/graph/data/mod_graph_{args.dataset}-0.7-0.1-0.1-0-dice-0.15/mod_adj.pt", map_location='cpu')

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
    ########################

    num_test = data.test_mask.sum().item()
    num_known_in_test = (data.unknown_in_unseen_mask & data.test_mask).sum().item()
    ratio = num_known_in_test / num_test if num_test > 0 else 0.0
    print(f"测试集总样本数：{num_test}")
    print(f"测试集中 Known-in-Unseen 样本数：{num_known_in_test}")
    print(f"Known-in-Unseen 占测试集的比例：{ratio:.4f}")
    #######################

    # compare_class_similarity(
    #     edge_index_clean=data.edge_index,
    #     edge_index_attacked=adj,
    #     y=data.y,
    #     # train_mask=data.train_mask,
    #     # known_mask=data.known_in_unseen_mask,
    #     # unknown_mask=data.unknown_in_unseen_mask,
    #     num_classes=data.y.max().item() + 1,
    #     mask=data.train_mask
    # )
    
    # compare_class_similarity(
    #     edge_index_clean=data.edge_index,
    #     edge_index_attacked=adj,
    #     y=data.y,
    #     # train_mask=data.train_mask,
    #     # known_mask=data.known_in_unseen_mask,
    #     # unknown_mask=data.unknown_in_unseen_mask,
    #     num_classes=data.y.max().item() + 1,
    #     mask=data.test_mask
    # )

    # data.edge_index = adj

    device = torch.device(args.device)
    # Initialize DGIBNN model
    args.nfeat = dataset.num_features
    args.nhid = args.nhid
    args.n_layers = args.n_layers
    args.nout = num_known_classes
    model = DGIBNN_mlt(args).to(device)
    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    patience_counter = 0
    best_val = 0.0
    best_pseudo_auroc_val = 0.0
    best_overall_acc_test = 0.0
    best_overall_f1_test = 0.0
    best_openset_metrics = None

    test_acc_list = []
    test_f1_list = []
    id_val_acc_list = []
    overal_val_acc_list = []

    test_known_acc_list = []
    test_known_f1_list = []
    test_ood_acc_list = []
    test_ood_f1_list = []

    ood_auroc_list = [] 
    pseu_ood_auroc_list = []
    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1)):
        loss = train(model, data, optimizer, device, args, epoch)
        accs = test(model, data, device, epoch)
        if accs['overall_val'] > best_val:
            best_val = accs['overall_val']
            best_overall_acc_test = accs['multi_class']['acc']
            best_overall_f1_test = accs['multi_class']['f1_macro']
            # torch.save({
            #             'epoch': epoch,
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'val_acc': best_val
            #             }, args.save_path)
        else:
            patience_counter += 1

        if accs['pseudo_openset'][0] > best_pseudo_auroc_val:
            best_openset_metrics = accs['openset']
            best_pseudo_auroc_val = accs['pseudo_openset'][0]

       # 在main函数的训练循环中修改打印语句
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
            print(f"Train: {accs['train']:.4f} | Val: {accs['val']:.4f}")
            print(f"Known Classes - Acc: {accs['multi_class']['known_acc']:.4f}, F1: {accs['multi_class']['known_f1']:.4f}")
            print(f"OOD Class - Acc: {accs['multi_class']['ood_acc']:.4f}, F1: {accs['multi_class']['ood_f1']:.4f}")
            print(f"Overall - Acc: {accs['multi_class']['acc']:.4f}, F1: {accs['multi_class']['f1_macro']:.4f}")
            print(f"OOD Detection - AUROC: {accs['openset'][0]:.4f}")
            print(f"Overall Val: {accs['overall_val']:.4f}") 

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
        
        # 在每个epoch记录各项指标
        test_acc_list.append(accs['multi_class']['acc'])
        test_f1_list.append(accs['multi_class']['f1_macro'])

        test_known_acc_list.append(accs['multi_class']['known_acc'])
        test_known_f1_list.append(accs['multi_class']['known_f1'])
        test_ood_acc_list.append(accs['multi_class']['ood_acc'])
        test_ood_f1_list.append(accs['multi_class']['ood_f1'])

        id_val_acc_list.append(accs['val'].cpu())
        overal_val_acc_list.append(accs['overall_val'].cpu())   
        ood_auroc_list.append(accs['openset'][0])
        

    # Load best model and evaluate
    print(f"\nBest validation/test overall accuracy/test f1: {best_val:.4f}/{best_overall_acc_test:.4f}/{best_overall_f1_test:.4f}, model saved to {args.save_path}")
    
    # checkpoint = torch.load(args.save_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # accs = test(model, data, device)
    # print("\n=== Test after loading best model ===")
    # print(f"Train: {accs['train']:.4f} | Val: {accs['val']:.4f} | Test: {accs['test']:.4f}")


    plt.figure()
    plt.plot(test_acc_list, label="ID Accuracy (All)")
    # plt.plot(test_f1_list, label="ID F1 Score (All)")

    plt.plot(id_val_acc_list, label="ID Accuracy (Val)")
    plt.plot(overal_val_acc_list, label="Overall Accuracy (Val)")


    plt.plot(test_known_acc_list, label="Known Classes Acc")
    # plt.plot(test_known_f1_list, label="Known Classes F1")
    plt.plot(test_ood_acc_list, label="OOD Class Acc")
    # plt.plot(test_ood_f1_list, label="OOD Class F1")
    # plt.plot(ood_auroc_list, label="OOD Detection AUROC")
    # plt.plot(pseu_ood_auroc_list, label="Pseudo OOD Detection AUROC")


    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.legend()
    plt.title("ID Classification vs OOD Detection")
    plt.grid(True)
    plt.savefig("id_vs_ood_performance.png")
    plt.show()

    # 绘制损失的图
    plt.figure()
    plt.plot(ixz, label="IXZ Loss", color='r')
    plt.plot(ia, label="IA Loss", color='g')
    plt.plot(i_both, label="IA + IXZ Loss", color='c')
    plt.plot(i_yz_list, label="I(Y;Z) Loss", color='y')
    plt.plot(i_yoodz_list, label="I(Y;Z) OOD Loss", color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Losses during Training")
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    main()

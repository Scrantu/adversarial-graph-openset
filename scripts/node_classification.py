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
from DGIB.model import DGIBNN


def train(model, classifier, data, optimizer, device, args):
    model.train()
    classifier.train()
    optimizer.zero_grad()
    # Static graph: single snapshot
    x_all = [data.x.to(device)]
    edge_index_all = [data.edge_index.to(device)]
    # Forward through DGIBNN
    embeddings_list, ixz_loss, struct_kl_loss, consensual_loss = model(x_all, edge_index_all)
    embeddings = embeddings_list[0]

    outs = F.log_softmax(embeddings, dim=1)
    # Compute cross-entropy loss on train mask
    y_true = data.y.to(device)
    loss_cls = F.nll_loss(outs[data.train_mask], y_true[data.train_mask])
    # Combine with DGIB losses
    loss = loss_cls \
           + args.lambda_ixz * ixz_loss \
           + args.lambda_struct * struct_kl_loss 
    loss.backward()
    optimizer.step()
    return loss.item()


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

@torch.no_grad()
def test(model, classifier, data, device):
    model.eval()
    classifier.eval()
    x_all = [data.x.to(device)]
    edge_index_all = [data.edge_index.to(device)]
    embeddings_list, _, _, _ = model(x_all, edge_index_all)
    logits = embeddings_list[0]
    outs = F.log_softmax(logits, dim=1)
    print('outs', outs)
    y_true = data.y.to(device)
    accs = accuracy(outs[data.test_mask], y_true[data.test_mask])
    return accs, accs, accs # [train_acc, val_acc, test_acc]


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
    parser.add_argument('--lambda_ixz', type=float, default=0.1, help='Weight for I(X;Z) loss')
    parser.add_argument('--lambda_struct', type=float, default=0.1, help='Weight for structure KL loss')
    parser.add_argument('--lambda_cons', type=float, default=0.0, help='Weight for consensual loss')
    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--use_RTE', type=bool, default=False)
    args = parser.parse_args()

    # Load Cora dataset
    dataset = Planetoid(root='../data/cora', name='cora')
    data = dataset[0]
 
    device = torch.device(args.device)
    # Initialize DGIBNN model
    args.nfeat = dataset.num_features
    args.nhid = args.nhid
    args.n_layers = args.n_layers
    args.nout = dataset.num_classes
    model = DGIBNN(args).to(device)
    # Classification head
    classifier = nn.Linear(args.nhid, dataset.num_classes).to(device)
    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        loss = train(model, classifier, data, optimizer, device, args)
        if epoch == 1 or epoch % 10 == 0:
            train_acc, val_acc, test_acc = test(model, classifier, data, device)
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} '  
                  f'| Val: {val_acc:.4f} | Test: {test_acc:.4f}')
    # Final evaluation
    train_acc, val_acc, test_acc = test(model, classifier, data, device)
    print('=== Final Results ===')
    print(f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')


if __name__ == '__main__':
    main()

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import nni
import random
import numpy as np
import json
import os

# 固定随机种子
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(F.log_softmax(out[data.train_mask], dim=1), data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out = F.log_softmax(model(data.x, data.edge_index), dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(accuracy(out[mask], data.y[mask]))
    return accs

def save_nni_configs():
    # 保存 search_space.json
    search_space = {
        "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        "dropout": {"_type": "uniform", "_value": [0.3, 0.8]},
        "nhid": {"_type": "choice", "_value": [64, 128, 256]}
    }
    with open("search_space.json", "w") as f:
        json.dump(search_space, f, indent=4)

    # 保存 config.yml
    config_yaml = """
experimentName: gcn_nni
trialConcurrency: 2
maxTrialNumber: 20
maxExperimentDuration: 1h

searchSpaceFile: search_space.json
trialCommand: python3 gcn_nni_train.py
trialCodeDirectory: .

tuner:
  name: TPE
  classArgs:
    optimize_mode: maximize
"""
    with open("config.yml", "w") as f:
        f.write(config_yaml)

    print("✅ 已生成 search_space.json 和 config.yml，可使用以下命令启动 NNI 实验：")
    print("\n    nnictl create --config config.yml\n")
    print("打开浏览器查看: http://localhost:8080")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_nni_config', type=bool, default=True)
    args = parser.parse_args()

    if args.save_nni_config:
        save_nni_configs()
        return

    try:
        nni_params = nni.get_next_parameter()
        args.lr = nni_params.get("lr", args.lr)
        args.dropout = nni_params.get("dropout", args.dropout)
        args.nhid = nni_params.get("nhid", args.nhid)
    except:
        print("NNI not in use. Using default args.")

    print(f"Using: lr={args.lr}, dropout={args.dropout}, nhid={args.nhid}")

    dataset = Planetoid(root="../data/cora", name="cora")
    data = dataset[0]

    model = GCN(dataset.num_features, args.nhid, dataset.num_classes, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = 0.0
    best_test = 0.0
    for epoch in range(args.epochs):
        train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Val {val_acc:.4f}, Test {test_acc:.4f}")

    nni.report_final_result(best_val.item())
    print(f"Best Val Acc: {best_val:.4f}, Test Acc: {best_test:.4f}")

if __name__ == '__main__':
    main()

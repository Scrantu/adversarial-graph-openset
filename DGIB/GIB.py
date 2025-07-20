import sys
import math
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
# 添加 DGIB 模块路径
sys.path.append('./')
sys.path.append('../')

from DGIB.pytorch_net.net import reparameterize, Mixture_Gaussian_reparam
from DGIB.pytorch_net.util import (
    sample, to_cpu_recur, to_np_array, to_Variable, record_data,
    make_dir, remove_duplicates, update_dict, get_list_elements,
    to_string, filter_filename
)
from DGIB.util_IB import (
    get_reparam_num_neurons, sample_lognormal, scatter_sample,
    uniform_prior, compose_log, edge_index_2_csr
)

# 设置随机种子

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops
import torch.nn as nn

class StaticGIBConv(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 heads=4, reparam_mode='diag', prior_mode='mixGau-5',
                 struct_dropout_mode=('categorical',), sample_size=1, nbsz=15, attweight_in_channels=None):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_neurons = get_reparam_num_neurons(out_channels, reparam_mode)
        self.reparam_mode = reparam_mode
        self.prior_mode = prior_mode
        self.struct_dropout_mode = struct_dropout_mode
        self.sample_size = sample_size
        self.nbsz = nbsz
        # output dim of this conv
        self.weight = nn.Parameter(torch.Tensor(attweight_in_channels, heads * self.out_neurons))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * self.out_neurons))
        if prior_mode.startswith('mixGau'):
            n_comp = int(prior_mode.split('-')[1])
            self.feature_prior = Mixture_Gaussian_reparam(
                is_reparam=False, Z_size=self.out_neurons * heads, n_components=n_comp
            )
        self.reset_parameters()
        self.ixz = None
        self.structure_kl = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        edge_index,_ = remove_self_loops(edge_index)
        edge_index,_ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x @ self.weight
        out = self.propagate(edge_index, x=x)
        flat = out.view(-1, self.heads * self.out_neurons)
        dist, _ = reparameterize(model=None, input=flat,
                                 mode=self.reparam_mode, size=self.out_neurons)
        Z_core = sample(dist, self.sample_size)
        if self.prior_mode.startswith('mixGau'):
            log_q = dist.log_prob(Z_core).sum(-1)
            prior_log = self.feature_prior.log_prob(Z_core).sum(-1)
            ixz = (log_q - prior_log).mean()
        else:
            prior = torch.distributions.Normal(
                loc=torch.zeros_like(dist.loc), scale=torch.ones_like(dist.scale)
            )
            ixz = torch.distributions.kl.kl_divergence(dist, prior).sum(-1).mean()
        self.ixz = ixz
        self.structure_kl = torch.tensor(0.0, device=x.device)
        out = Z_core.mean(0).view(-1, self.heads * self.out_neurons)
        return out

    def message(self, x_j, x_i, edge_index_i, size_i):
        E = x_j.size(0)
        x_j_h = x_j.view(E, self.heads, self.out_neurons)
        x_i_h = x_i.view(E, self.heads, self.out_neurons)
        alpha = (torch.cat([x_i_h, x_j_h], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        prior = uniform_prior(edge_index_i)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha_sampled = scatter_sample(alpha, edge_index_i,
                                       self.struct_dropout_mode[0], size_i)
        kl = (alpha * (torch.log(alpha + 1e-12) - torch.log(prior + 1e-12))).sum()
        self.structure_kl = kl / alpha.numel()
        return (x_j_h * alpha_sampled.unsqueeze(-1)).view(E, self.heads * self.out_neurons)

    def update(self, aggr_out):
        return aggr_out

class StaticGIBGAT(nn.Module):
    def __init__(self, in_channels, hid, out):
        super().__init__()
        self.conv1 = StaticGIBConv(in_channels, hid, heads=4, attweight_in_channels=in_channels)
        self.conv2 = StaticGIBConv(hid * 4, hid, heads=1, attweight_in_channels=1024)
        self.fc = nn.Linear(hid*2, out) # 2要自己设置

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        sib1, xib1 = self.conv1.structure_kl, self.conv1.ixz
        x = F.dropout(x, 0.5, self.training)
        x = F.elu(self.conv2(x, edge_index))
        sib2, xib2 = self.conv2.structure_kl, self.conv2.ixz
        x = F.dropout(x, 0.5, self.training)
        logits = self.fc(x)
        sib = sib1 + sib2
        xib = xib1 + xib2
        return logits, sib, xib

@torch.no_grad()
def test(model, data):
    model.eval()
    logits, _, _ = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    accs=[]
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append((pred[mask]==data.y[mask]).float().mean().item())
    return accs

def main():
    set_seed(42)
    ds = Planetoid(root='/Users/scrantu/Desktop/code/GIB/data/Cora', name='Cora', transform=NormalizeFeatures())
    data = ds[0]
    model = StaticGIBGAT(ds.num_features, 128, ds.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(1,201):
        model.train(); optimizer.zero_grad()
        logits, sib, xib = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask]) + 0.01*(sib + xib)
        loss.backward(); optimizer.step()
        if epoch % 20 == 0:
            t,v,tst = test(model, data)
            print(f"Epoch {epoch}, Loss {loss:.4f}, Train {t:.4f}, Val {v:.4f}, Test {tst:.4f}")

if __name__=='__main__': main()

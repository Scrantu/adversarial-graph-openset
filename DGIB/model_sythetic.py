from torch_geometric.nn.inits import glorot, zeros
from torch import nn
from torch.nn import Parameter
from torch_geometric.utils import (
    add_remaining_self_loops,
    remove_self_loops,
    add_self_loops,
    softmax,
    degree,
    to_undirected,
)
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn.conv import MessagePassing
from torch.distributions.normal import Normal
from torch_scatter import scatter
from collections import OrderedDict
import torch.nn.functional as F
import networkx as nx
import numpy as np
import torch
import math
from torch_sparse import SparseTensor, matmul
from tqdm import tqdm

from DGIB.pytorch_net.net import reparameterize, Mixture_Gaussian_reparam
from DGIB.pytorch_net.util import (
    sample,
    to_cpu_recur,
    to_np_array,
    to_Variable,
    record_data,
    make_dir,
    remove_duplicates,
    update_dict,
    get_list_elements,
    to_string,
    filter_filename,
)
from DGIB.util_IB import (
    get_reparam_num_neurons,
    sample_lognormal,
    scatter_sample,
    uniform_prior,
    compose_log,
    edge_index_2_csr,
    COLOR_LIST,
    LINESTYLE_LIST,
    process_data_for_nettack,
    parse_filename,
    add_distant_neighbors,
)

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def visualize_energy_distributions(neg_energy_ind, neg_energy_openset, title="Energy Distribution"):
    """可视化 IND 和 Open-set 节点的能量分布"""
    neg_energy_ind = neg_energy_ind.detach().cpu().numpy()
    neg_energy_openset = neg_energy_openset.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    sns.kdeplot(neg_energy_ind, label='Known Classes (IND)', fill=True, color='blue', linewidth=2)
    sns.kdeplot(neg_energy_openset, label='Unknown Classes (Open-set)', fill=True, color='red', linewidth=2)
    plt.xlabel("Score (Negative Energy)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



class DGIBConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        reparam_mode=None,
        prior_mode=None,
        distribution=None,
        temperature=0.2,
        nbsz=15,
        val_use_mean=True,
        sample_size=1,
        bias=True,
        agg_param=0.8,
        **kwargs
    ):
        super(DGIBConv, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        self.out_neurons = out_channels
        self.distribution = distribution
        self.temperature = temperature
        self.nbsz = nbsz
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean
        self.agg_param = agg_param
        self.weight = Parameter(torch.Tensor(in_channels, heads * self.out_neurons))
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_neurons))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_neurons))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter("bias", None)

        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(
                    is_reparam=False,
                    Z_size=self.out_channels,
                    n_components=n_components,
                )

        self.skip_editing_edge_index = False
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward_single_graphshot(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x) and not self.skip_editing_edge_index:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (
                None if x[0] is None else torch.matmul(x[0], self.weight),
                None if x[1] is None else torch.matmul(x[1], self.weight),
            )

        out = self.propagate(edge_index, size=size, x=x)

        if self.reparam_mode is not None:
            # Reparameterize:
            out = out.view(-1, self.out_neurons)
            self.dist, _ = reparameterize(
                model=None, input=out, mode=self.reparam_mode, size=self.out_channels,
            )  # dist: [B * head, Z]
            Z_core = sample(self.dist, self.sample_size)  # [S, B * head, Z]
            Z = Z_core.view(
                self.sample_size, -1, self.heads * self.out_channels
            )  # [S, B, head * Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(
                    loc=torch.zeros(out.size(0), self.out_channels).to(x.device),
                    scale=torch.ones(out.size(0), self.out_channels).to(x.device),
                )  # feature_prior: [B * head, Z]

            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = (
                    torch.distributions.kl.kl_divergence(self.dist, self.feature_prior)
                    .sum(-1)
                    .view(-1, self.heads)
                    .mean(-1)
                )
            else:
                Z_logit = (
                    self.dist.log_prob(Z_core).sum(-1)
                    if self.reparam_mode.startswith("diag")
                    else self.dist.log_prob(Z_core)
                )  # [S, B * head]
                prior_logit = self.feature_prior.log_prob(Z_core).sum(
                    -1
                )  # [S, B * head]
                # upper bound of I(X; Z):
                ixz = (
                    (Z_logit - prior_logit).mean(0).view(-1, self.heads).mean(-1)
                )  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)
            else:
                out = (
                    out[:, : self.out_channels]
                    .contiguous()
                    .view(-1, self.heads * self.out_channels)
                )
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)

        if "categorical" in self.distribution:
            structure_kl_loss = torch.sum(
                self.alpha * torch.log((self.alpha + 1e-16) / self.prior)
            )
        elif "Bernoulli" in self.distribution:
            posterior = torch.distributions.bernoulli.Bernoulli(self.alpha)
            prior = torch.distributions.bernoulli.Bernoulli(logits=self.prior)
            structure_kl_loss = (
                torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
            )
        else:
            raise Exception(
                "I think this belongs to the diff subset sampling that is not implemented"
            )

        return out, torch.mean(ixz), structure_kl_loss

    def forward(self, x_all, edge_index_all, size=None):
        times = len(x_all)
        dev = x_all[0].device
        n = len(x_all[0])
        out_list = torch.zeros((times, x_all[0].size(0), self.out_channels)).to(dev)
        self.ixz_list = [[] for _ in range(times)]
        self.structure_kl_loss_list = [[] for _ in range(times)]

        for t in range(times):
            (
                out,
                self.ixz_list[t],
                self.structure_kl_loss_list[t],
            ) = self.forward_single_graphshot(x_all[t], edge_index_all[t])
            if t > 0:
                weights = F.sigmoid(
                    torch.tensor(list(range(t))).view(t, 1, 1).to(x_all[0].device)
                )
                out_list[t] = (
                    torch.sum(weights * out_list[:t], dim=0) / t
                ) * self.agg_param + out * (1 - self.agg_param)
            else:
                out_list[t] = out
        ixz_mean = torch.mean(torch.stack(self.ixz_list))
        structure_kl_loss_mean = torch.mean(torch.stack(self.structure_kl_loss_list))

        return out_list, ixz_mean, structure_kl_loss_mean, self.ixz_list[t - 1]

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(
            -1, self.heads, self.out_neurons
        )  # [N_edge, heads, out_channels]
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_neurons :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_neurons)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(
                dim=-1
            )  # [N_edge, heads]

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Sample attention coefficients stochastically.
        neighbor_sampling_mode = self.distribution
        if "categorical" in neighbor_sampling_mode:
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            self.alpha = alpha
            self.prior = uniform_prior(edge_index_i)
            if self.val_use_mean is False or self.training:
                temperature = self.temperature
                sample_neighbor_size = self.nbsz
                if neighbor_sampling_mode == "categorical":
                    alpha = scatter_sample(alpha, edge_index_i, temperature, size_i)
                elif "multi-categorical" in neighbor_sampling_mode:
                    alphas = []
                    for _ in range(
                        sample_neighbor_size
                    ):  #! this can be improved by parallel sampling
                        alphas.append(
                            scatter_sample(alpha, edge_index_i, temperature, size_i)
                        )
                    alphas = torch.stack(alphas, dim=0)
                    if "sum" in neighbor_sampling_mode:
                        alpha = alphas.sum(dim=0)
                    elif "max" in neighbor_sampling_mode:
                        alpha, _ = torch.max(alphas, dim=0)
                    else:
                        raise
                else:
                    raise
        elif neighbor_sampling_mode == "Bernoulli":
            alpha_normalization = torch.ones_like(alpha)
            alpha_normalization = softmax(
                alpha_normalization, edge_index_i, num_nodes=size_i
            )
            alpha = torch.clamp(torch.sigmoid(alpha), 0.01, 0.99)
            self.alpha = alpha
            self.prior = (torch.ones_like(self.alpha) * (1 / self.nbsz)).to(
                alpha.device
            )
            if not self.val_use_mean or self.training:
                temperature = self.temperature
                alpha = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                    torch.Tensor([temperature]).to(alpha.device), probs=alpha
                ).rsample()
            alpha = alpha * alpha_normalization
        else:
            raise

        return (x_j * alpha.view(-1, self.heads, 1)).view(
            -1, self.heads * self.out_neurons
        )

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_neurons)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def to_device(self, device):
        self.to(device)
        return self

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class DGIBNN(torch.nn.Module):
    def __init__(self, args=None):

        super(DGIBNN, self).__init__()
        self.num_features = args.nfeat
        self.normalize = True
        self.reparam_mode = args.reparam_mode
        self.prior_mode = args.prior_mode  #
        self.distribution = args.distribution
        self.temperature = args.temperature
        self.nbsz = args.nbsz
        self.dropout = args.dropout
        self.latent_size = args.nhid
        self.out_size = args.nout
        self.sample_size = args.sample_size
        self.num_layers = args.n_layers
        self.with_relu = True
        self.val_use_mean = True
        self.reparam_all_layers = True
        self.is_cuda = True
        self.device = torch.device(
            self.is_cuda
            if isinstance(self.is_cuda, str)
            else "cuda"
            if self.is_cuda
            else "cpu"
        )
        self.ixz = 0
        self.structure_kl_loss = 0
        self.consensual = 0
        self.args = args

        self.init()

    def init(self):
        self.reparam_layers = []
        latent_size = self.latent_size

        for i in range(self.num_layers):
            if i == 0:
                input_size = self.num_features
            else:
                input_size = latent_size
            if self.reparam_all_layers is True:
                is_reparam = True
            elif isinstance(self.reparam_all_layers, tuple):
                reparam_all_layers = tuple(
                    [
                        kk + self.num_layers if kk < 0 else kk
                        for kk in self.reparam_all_layers
                    ]
                )
                is_reparam = i in reparam_all_layers
            else:
                raise
            if is_reparam:
                self.reparam_layers.append(i)

            if i == self.num_layers - 1:
                latent_size = self.out_size
            setattr(
                self,
                "conv{}".format(i + 1),
                DGIBConv(
                    input_size,
                    latent_size,
                    heads=self.args.heads if i != self.num_layers - 1 else 1,
                    concat=True,
                    reparam_mode=self.reparam_mode if is_reparam else None,
                    prior_mode=self.prior_mode if is_reparam else None,
                    distribution=self.distribution,
                    temperature=self.temperature,
                    nbsz=self.nbsz,
                    val_use_mean=self.val_use_mean,
                    sample_size=self.sample_size,
                ),
            )

        self.reparam_layers = sorted(self.reparam_layers)

        self.reg_params = self.parameters()
        self.non_reg_params = OrderedDict()  ##
        self.to(self.device)

    def to_device(self, device):
        for i in range(self.num_layers):
            getattr(self, "conv{}".format(i + 1)).to_device(device)
        self.to(device)
        return self

    def forward(self, x_all, edge_index_all):
        times = len(x_all)
        ixz, structure_kl_loss = [], []
        for i in range(self.num_layers):
            layer = getattr(self, "conv{}".format(i + 1))
            x_all, ixz_mean, structure_kl_loss_mean, ixz_cons = layer(
                x_all, edge_index_all
            )
            ixz.append(ixz_mean)
            structure_kl_loss.append(structure_kl_loss_mean)

            x_all = [F.elu(x) for x in x_all]
            x_all = [
                F.dropout(
                    input=F.normalize(x, dim=-1), p=self.dropout, training=self.training
                )
                for x in x_all
            ]
        return (
            x_all,
            torch.stack(ixz).mean(),
            torch.stack(structure_kl_loss).mean(),
        )
    
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


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def intermediate_forward(self, x, edge_index=None, layer_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature_list(self, x, edge_index=None):
        out_list = []
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out_list.append(x)
        x = self.lins[-1](x)
        return x, out_list
    

class DGIBNN_mlt(torch.nn.Module):
    def __init__(self, args=None):
        super(DGIBNN_mlt, self).__init__()
        # 基本属性
        self.num_features = args.nfeat
        self.latent_size  = args.nhid
        self.out_size     = args.nout
        self.reparam_mode = args.reparam_mode
        self.prior_mode   = args.prior_mode
        self.distribution = args.distribution
        self.temperature  = args.temperature
        self.nbsz         = args.nbsz
        self.sample_size  = args.sample_size
        self.dropout      = args.dropout
        self.args         = args

        # 生成伪样本超参
        self.coef_reg       = 1
        self.mcmc_steps     = 20
        self.mcmc_step_size = 1
        self.mcmc_noise     = 0.005
        self.max_buffer_vol = 2
        self.buffer_prob = 0.95
        self.replay_buffer  = None
        self.p = None
        self.c = None
        # 能量网络（采用双线性或 MLP，根据需要）
        self.energy_net = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size, 1)
        )

        self.d_phi = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, self.latent_size//2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_size//2, 1),
            torch.nn.Sigmoid()
        )

        # 共享第一层
        self.shared_conv = DGIBConv(
            in_channels=self.num_features,
            out_channels=self.latent_size,
            heads=args.heads,
            concat=True,
            reparam_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            distribution=self.distribution,
            temperature=self.temperature,
            nbsz=self.nbsz,
            val_use_mean=True,
            sample_size=self.sample_size,
        )

        # 支路1第二层
        self.branch1_conv = DGIBConv(
            in_channels=self.latent_size,
            out_channels=self.latent_size,
            heads=1,
            concat=True,
            reparam_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            distribution=self.distribution,
            temperature=self.temperature,
            nbsz=self.nbsz,
            val_use_mean=True,
            sample_size=self.sample_size,
        )

        # 支路2第二层
        self.branch2_conv = DGIBConv(
            in_channels=self.latent_size,
            out_channels=self.latent_size,
            heads=1,
            concat=True,
            reparam_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            distribution=self.distribution,
            temperature=self.temperature,
            nbsz=self.nbsz,
            val_use_mean=True,
            sample_size=self.sample_size,
        )
        self.branch1_cls = nn.Linear(self.latent_size, self.out_size)
        # 移动到设备
        self.to(self._get_device())

    def _get_device(self):
        return torch.device(self.args.device)

    def forward(self, x_all, edge_index_all):
        ixz_all = []
        structure_kl_loss = []

        # 共享层前向
        x0_list, shared_ixz, shared_kl, _ = self.shared_conv(x_all, edge_index_all)
        x0 = x0_list[0]
        x0 = F.elu(x0)
        x0 = F.dropout(F.normalize(x0, dim=-1), p=self.dropout, training=self.training)
        ixz_all.append(shared_ixz)
        structure_kl_loss.append(shared_kl)
        # 构造支路输入
        branch_input = [x0]

        # 支路1前向
        x1_list, ixz1_mean, kl1, _ = self.branch1_conv(branch_input, edge_index_all)
        x1 = x1_list[0]
        x1 = F.elu(x1)
        x1 = F.dropout(F.normalize(x1, dim=-1), p=self.dropout, training=self.training)
        ixz_all.append(ixz1_mean)
        structure_kl_loss.append(kl1)
        
        logits = self.branch1_cls(x1)
        # x_concat = torch.cat([x0, x1], dim=-1)

        return [x1], [logits], torch.stack(ixz_all).mean(), torch.stack(structure_kl_loss).mean()
    
    def energy_gradient(self, x):
        self.energy_net.eval()  # 如果你还有 encoder，也加 encoder.eval()
        self.shared_conv.eval()
        self.branch1_conv.eval()
        self.branch2_conv.eval()
        self.branch1_cls.eval()
        self.d_phi.eval()

        x_i = x.clone().detach().requires_grad_(True)
        x_i = F.elu(x_i)
        x_i = F.dropout(F.normalize(x_i, dim=-1), p=0.5, training=True)

        e = self.energy_net(x_i).sum()
        # grad = torch.autograd.grad(e, x_i)[0]
        grad = torch.autograd.grad(e, [x_i], retain_graph=True)[0]

        self.energy_net.train()
        self.shared_conv.train()
        self.branch1_conv.train()
        self.branch2_conv.train()
        self.branch1_cls.train()
        self.d_phi.train()

        return grad

    def langevin_dynamics_step(self, x_old):
        grad = self.energy_gradient(x_old)
        noise = torch.randn_like(grad) * self.mcmc_noise
        x_new = x_old + self.mcmc_step_size * grad + noise
        return x_new.detach()

    def langevin_dynamics_step_with_weighted_alignment(self, x_old, x_cand, weights_cand, lambda_align=0.5):
        grad = self.energy_gradient(x_old)  # [B, D]
        noise = torch.randn_like(x_old) * self.mcmc_noise  # [B, D]

        if x_cand is not None and x_cand.size(0) > 0:
            B = x_old.size(0)
            N = x_cand.size(0)

            x_old_exp = x_old.unsqueeze(1).expand(B, N, -1)    # [B, N, D]
            x_cand_exp = x_cand.unsqueeze(0).expand(B, N, -1)  # [B, N, D]
            weights_exp = weights_cand.unsqueeze(0).expand(B, N, -1)  # [B, N, 1]

            diff = x_cand_exp - x_old_exp  # [B, N, D]
            weighted_diff = (diff * weights_exp).sum(dim=1) / weights_exp.sum(dim=1)  # [B, D]
        else:
            weighted_diff = 0.0

        x_new = lambda_align * (x_old - 0.5 * self.mcmc_step_size * grad + noise) \
            + (1 - lambda_align) * weighted_diff

        return x_new.detach()



    def sample(self, sample_size, max_buffer_len, device, x_cand=None, weights_cand=None):
        if self.replay_buffer is None:
            x = 2.0 * torch.rand(sample_size, self.latent_size, device=device) - 1.0
        else:
            idx = torch.randperm(self.replay_buffer.size(0))[:int(sample_size * self.buffer_prob)]
            x = torch.cat([
                self.replay_buffer[idx],
                2.0 * torch.rand(sample_size - idx.size(0), self.latent_size, device=device) - 1.0
            ], dim=0)

        for _ in range(self.mcmc_steps):
            x = self.langevin_dynamics_step_with_weighted_alignment(
                x.to(device), x_cand=x_cand, weights_cand=weights_cand
            )

        if self.replay_buffer is None:
            self.replay_buffer = x
        else:
            self.replay_buffer = torch.cat([x, self.replay_buffer], dim=0)[:max_buffer_len]

        return x


    # def sample(self, sample_size, max_buffer_len, device):
    #     # 初始化或从缓冲区采样
    #     if self.replay_buffer is None:
    #         x = 2.0 * torch.rand(sample_size, self.latent_size, device=device) - 1.0
    #     else:
    #         idx = torch.randperm(self.replay_buffer.size(0))[:int(sample_size * self.buffer_prob)]
    #         x = torch.cat([
    #             self.replay_buffer[idx],
    #             2.0 * torch.rand(sample_size - idx.size(0), self.latent_size, device=device) - 1.0
    #         ], dim=0)
    #     # MCMC 采样
    #     for _ in range(self.mcmc_steps):
    #         x = self.langevin_dynamics_step(x.to(device))
    #     # 更新缓冲区
    #     if self.replay_buffer is None:
    #         self.replay_buffer = x
    #     else:
    #         self.replay_buffer = torch.cat([x, self.replay_buffer], dim=0)[:max_buffer_len]
    #     return x

    def gen_loss(self, energy_pos, energy_neg, reduction='mean'):
        # 生成损失: 拉开正（ID）与伪（sample）能量
        loss = energy_pos - energy_neg
        loss_reg = energy_pos.pow(2) + energy_neg.pow(2)
        loss = loss + self.coef_reg * loss_reg
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def pu_discriminator_loss(self, x_labeled, x_unlabeled, mu=0.5, reduction='mean'):
        d_phi = self.d_phi

        out_pos = torch.clamp(d_phi(x_labeled), 1e-6, 1 - 1e-6)
        out_unl = torch.clamp(d_phi(x_unlabeled), 1e-6, 1 - 1e-6)

        loss_pos_1 = F.binary_cross_entropy(out_pos, torch.ones_like(out_pos), reduction='none')
        loss_pos_0 = F.binary_cross_entropy(out_pos, torch.zeros_like(out_pos), reduction='none')
        loss_unl_0 = F.binary_cross_entropy(out_unl, torch.zeros_like(out_unl), reduction='none')

        if reduction == 'mean':
            pu_loss = mu * loss_pos_1.mean() - mu * loss_pos_0.mean() + loss_unl_0.mean()
        elif reduction == 'sum':
            pu_loss = mu * loss_pos_1.sum() - mu * loss_pos_0.sum() + loss_unl_0.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

        return pu_loss

    def compute_openset_weight(self, d_probs, mu=0.5):
        """
        输入:
            d_probs: 判别器输出 d(x)，应在 [0,1]，表示为 ID 的概率
            mu: unlabeled 样本中 ID 样本的先验比例（如 0.3）

        输出:
            w_ood: 每个样本属于 OOD 的重要性权重
        """
        d = torch.clamp(d_probs, min=1e-6, max=1-1e-6)  # 避免除0
        numerator = d
        denominator = mu * d + (1 - mu) * (1 - d)
        w_id = numerator / denominator
        w_ood = 1.0 - w_id
        return w_ood

    def select_topk_ood_candidates(self, weights: torch.Tensor, mu: float, unlabeled_mask: torch.Tensor) -> torch.Tensor:
        """
        选择 unlabeled 中 OOD 权重 Top-(1 - mu) 的节点索引（Bool Mask）

        参数:
            weights: [N]，全图节点的 OOD 权重
            mu: 正类比例
            unlabeled_mask: [N]，布尔张量，True 表示 unlabeled 节点

        返回:
            mask_cand: [N]，布尔张量，True 表示选中的伪 OOD 节点
        """
        weights_unlabeled = weights[unlabeled_mask]  # 提取 unlabeled 部分
        N_unl = weights_unlabeled.size(0)
        k = int((1 - mu) * N_unl)

        if k == 0 or N_unl == 0:
            return torch.zeros_like(weights, dtype=torch.bool)  # 返回全 False

        # 获取阈值
        sorted_weights, _ = torch.sort(weights_unlabeled, descending=True)
        thresh = sorted_weights[k - 1].item()

        # 构造原始空间上的 mask
        mask_cand = torch.zeros_like(weights, dtype=torch.bool)
        mask_cand[unlabeled_mask] = weights_unlabeled >= thresh
        return mask_cand.view(-1)


    def detect(self, e, data, T=1.0, cos_threshold=-1, prop_layers=2, alpha=0.5):
        """
        基于余弦相似度剪枝邻接后，再进行能量传播。
        cos_threshold: 保留余弦相似度大于该值的边
        """
        # 1. 初始能量
        # e = T * torch.logsumexp(logits / T, dim=-1)

        # # 3. 传播
        # e = self.propagation(e, data.edge_index,
        #                      prop_layers=prop_layers, alpha=alpha)

        # 4. 拆分 ID 与 openset
        ind_idx, openset_idx = data.known_in_unseen_mask, data.unknown_in_unseen_mask
        neg_energy_ind = e[ind_idx]
        neg_energy_openset = e[openset_idx]
        return neg_energy_ind, neg_energy_openset
    
    def detect_logit(self, logits, data, T=1.0, cos_threshold=-1, prop_layers=2, alpha=0.5):
        """
        基于余弦相似度剪枝邻接后，再进行能量传播。
        cos_threshold: 保留余弦相似度大于该值的边
        """
        # 1. 初始能量
        e = T * torch.logsumexp(logits / T, dim=-1)

        # 3. 传播
        e = self.propagation(e, data.edge_index,
                             prop_layers=prop_layers, alpha=alpha)

        # 4. 拆分 ID 与 openset
        ind_idx, openset_idx = data.known_in_unseen_mask, data.unknown_in_unseen_mask
        neg_energy_ind = e[ind_idx]
        neg_energy_openset = e[openset_idx]
        return neg_energy_ind, neg_energy_openset
    
    def propagation(self, e, edge_index, prop_layers=2, alpha=0.5):
        """能量传播，返回传播后的标量能量 e"""
        # e: [N]
        e = e.unsqueeze(1)  # [N,1]
        N = e.size(0)
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.nan_to_num(d_norm, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)


import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings_tsne(z_all, z_sample, data, title="t-SNE Visualization of Embeddings"):
    """
    可视化表征：训练ID、测试ID、测试OOD 和 伪OOD采样。

    参数:
        z_all: torch.Tensor, 所有节点的嵌入表示（来自 encoder 输出）
        z_sample: torch.Tensor, 采样生成的伪OOD嵌入
        data: PyG 数据对象，需包含 train_mask、known_in_unseen_mask、unknown_in_unseen_mask 掩码
        title: str, 图标题
    """
    z_id_train = z_all[data.train_mask].detach().cpu()
    z_id_test = z_all[data.known_in_unseen_mask].detach().cpu()
    z_ood_test = z_all[data.unknown_in_unseen_mask].detach().cpu()
    z_fake_ood = z_sample.detach().cpu()

    z_all_vis = torch.cat([z_id_train, z_id_test, z_ood_test, z_fake_ood], dim=0)
    labels = (
        ['Train ID'] * len(z_id_train) +
        ['Test Known ID'] * len(z_id_test) +
        ['Test Unknown OOD'] * len(z_ood_test) +
        ['Fake OOD (Sample)'] * len(z_fake_ood)
    )

    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(z_all_vis.numpy())

    plt.figure(figsize=(8, 6))
    colors = {
        'Train ID': 'green',
        'Test Known ID': 'blue',
        'Test Unknown OOD': 'red',
        'Fake OOD (Sample)': 'black'
    }

    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(z_tsne[idx, 0], z_tsne[idx, 1], s=20, label=label, alpha=0.6, c=colors[label])

    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_energy_distributions(
    energy_ind: torch.Tensor,
    energy_sample: torch.Tensor,
    title: str = "Energy Distribution",
    save_path: str = None,
    bins: int = 50
):
    """
    可视化 ID 和伪 OOD 样本的能量分布（直方图 + KDE）。

    参数:
        energy_ind: [N] 张量，训练集中 ID 节点的能量
        energy_sample: [M] 张量，伪 OOD 节点的能量
        title: 图标题
        save_path: 如果提供路径则保存为图片，否则直接展示
        bins: 直方图 bin 数量
    """
    energy_ind = energy_ind.detach().cpu().numpy()
    energy_sample = energy_sample.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    sns.histplot(energy_ind, bins=bins, kde=True, stat='density',
                 label="Train (ID)", color='green', alpha=0.6)
    sns.histplot(energy_sample, bins=bins, kde=True, stat='density',
                 label="Sampled (pseudo OOD)", color='red', alpha=0.6)
    
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

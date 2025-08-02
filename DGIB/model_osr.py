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
        
        self.coef_reg       = 1
        self.mcmc_steps     = 20
        self.mcmc_step_size = 1
        self.mcmc_noise     = 0.005
        self.max_buffer_vol = 2
        self.buffer_prob = 0.95
        self.replay_buffer  = None
        self.p = None
        self.c = None
        
        self.d_phi = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, self.latent_size//2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_size//2, 1),
            torch.nn.Sigmoid()
        )
        self.energy_net = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size, 1)
        )
        self.branch1_cls = nn.Linear(self.latent_size, self.out_size)

        # 移动到设备
        self.to(self._get_device())

    def _get_device(self):
        return torch.device(self.args.device)

    def forward(self, x_all, edge_index_all):
        ixz = []
        structure_kl_loss = []
        # 共享层前向
        x_shared_list, shared_ixz, shared_kl, _ = self.shared_conv(x_all, edge_index_all)
        x_shared = x_shared_list[0]
        x_shared = F.elu(x_shared)
        x_shared = F.dropout(F.normalize(x_shared, dim=-1), p=self.dropout, training=self.training)
        ixz.append(shared_ixz)
        structure_kl_loss.append(shared_kl)
        # 构造支路输入
        branch_input = [x_shared]

        # 支路1前向
        x1_list, ixz1, kl1, _ = self.branch1_conv(branch_input, edge_index_all)
        x1 = x1_list[0]
        x1 = F.elu(x1)
        x1 = F.dropout(F.normalize(x1, dim=-1), p=self.dropout, training=self.training)
        ixz.append(ixz1)
        structure_kl_loss.append(kl1)

        x1 = self.branch1_cls(x1)
        # 支路2前向
        # x2_list, ixz2, kl2, _ = self.branch2_conv(branch_input, edge_index_all)
        # x2 = x2_list[0]
        # x2 = F.elu(x2)
        # x2 = F.dropout(F.normalize(x2, dim=-1), p=self.dropout, training=self.training)

        # # 返回每个分支的节点向量及其各自的 ixz_mean, kl_mean
        # x_branches   = [x1, x2]
        # ixz_means    = [ixz1, ixz2]
        # kl_means     = [kl1, kl2]
        return [x1_list[0]], [x1], torch.stack(ixz).mean(), torch.stack(structure_kl_loss).mean()


        
    def energy_gradient(self, x):
        self.energy_net.eval()  # 如果你还有 encoder，也加 encoder.eval()
        self.shared_conv.eval()
        self.branch1_conv.eval()
        self.branch1_cls.eval()
        self.d_phi.eval()

        x_i = x.clone().detach().requires_grad_(True)
        x_i = F.elu(x_i)
        x_i = F.normalize(x_i, dim=-1)
        logits_neg = self.branch1_cls(x_i)
        energy_neg = - torch.logsumexp(logits_neg, dim=-1).sum()
        # grad = torch.autograd.grad(e, x_i)[0]
        grad = torch.autograd.grad(energy_neg, [x_i], retain_graph=True)[0]

        self.energy_net.train()
        self.shared_conv.train()
        self.branch1_conv.train()
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

    def sample(self, sample_size, max_buffer_len, device, x_cand=None, weights_cand=None, lambda_align=0.5, x_init=None):
        if x_init is not None:
            x = x_init.detach().clone().to(device)  # 使用伪 ID 表征作为初始向量
        elif self.replay_buffer is not None:
            idx = torch.randperm(self.replay_buffer.size(0))[:int(sample_size * self.buffer_prob)]
            x = torch.cat([
                self.replay_buffer[idx],
                2.0 * torch.rand(sample_size - idx.size(0), self.latent_size, device=device) - 1.0
            ], dim=0)
        else:
            x = 2.0 * torch.rand(sample_size, self.latent_size, device=device) - 1.0

        for _ in range(self.mcmc_steps):
            x = self.langevin_dynamics_step_with_weighted_alignment(
                x.to(device), x_cand=x_cand, weights_cand=weights_cand, lambda_align=lambda_align
            )

        if self.replay_buffer is None:
            self.replay_buffer = x
        else:
            self.replay_buffer = torch.cat([x, self.replay_buffer], dim=0)[:max_buffer_len]

        return x
    
    def langevin_dynamics_step_id(self, x_old):
        grad = self.energy_gradient(x_old)
        noise = torch.randn_like(grad) * self.mcmc_noise
        x_new = x_old + self.mcmc_step_size * grad + noise
        return x_new.detach()

    def langevin_dynamics_step_with_weighted_alignment_id(self, x_old, x_cand, weights_cand, lambda_align=0.5):
        grad = -self.energy_gradient_id(x_old)  # [B, D]
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
    
    def sample_pseudo_id(self, sample_size, max_buffer_len, device, x_cand=None, weights_cand=None, lambda_align=0.5, x_init=None):
        if x_init is not None:
            x = x_init.detach().clone().to(device)  # 使用伪 ID 表征作为初始向量
        elif self.replay_buffer is not None:
            idx = torch.randperm(self.replay_buffer.size(0))[:int(sample_size * self.buffer_prob)]
            x = torch.cat([
                self.replay_buffer[idx],
                2.0 * torch.rand(sample_size - idx.size(0), self.latent_size, device=device) - 1.0
            ], dim=0)
        else:
            x = 2.0 * torch.rand(sample_size, self.latent_size, device=device) - 1.0

        for _ in range(self.mcmc_steps):
            x = self.langevin_dynamics_step_with_weighted_alignment_id(
                x.to(device), x_cand=x_cand, weights_cand=weights_cand, lambda_align=lambda_align
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

    # def pu_discriminator_loss(self, x_labeled, x_unlabeled, mu=0.5, reduction='mean'):
    #     # 使用 d_phi 计算正样本和未标记样本的预测概率
    #     out_pos = torch.clamp(self.d_phi(x_labeled.detach()), 1e-6, 1 - 1e-6)  # 正样本的预测概率
    #     out_unl = torch.clamp(self.d_phi(x_unlabeled.detach()), 1e-6, 1 - 1e-6)  # 未标记样本的预测概率

    #     # 正样本损失（通过sigmoid计算概率）
    #     loss_pos = -torch.log(out_pos)  # 计算正样本的损失
        
    #     # 反向正样本损失（通过sigmoid计算反向概率）
    #     loss_pos_inv = -torch.log(1 - out_pos)  # 计算反向正样本的损失
        
    #     # 未标记样本损失（通过sigmoid计算负样本的概率）
    #     loss_unl = -torch.log(1 - out_unl)  # 假设未标记样本作为负类，计算其损失

    #     # 对正样本和未标记样本进行加权
    #     if reduction == 'mean':
    #         loss_pos = loss_pos.mean()
    #         loss_pos_inv = loss_pos_inv.mean()
    #         loss_unl = loss_unl.mean()
    #     elif reduction == 'sum':
    #         loss_pos = loss_pos.sum()
    #         loss_pos_inv = loss_pos_inv.sum()
    #         loss_unl = loss_unl.sum()
    #     else:
    #         raise ValueError("reduction must be 'mean' or 'sum'")

    #     # 正负样本的损失加权
    #     positive_risk = mu * loss_pos  # 正样本损失加权
    #     negative_risk = -mu * loss_pos_inv + loss_unl  # 未标记样本损失加权

    #     # 确保负样本风险非负
    #     negative_risk = torch.clamp(negative_risk, min=0.0)  # 负样本损失不能为负值

    #     # 最终的PU损失
    #     pu_loss = positive_risk + negative_risk

    #     return pu_loss

    def pu_discriminator_loss(self, x_labeled, x_unlabeled, mu=0.5, reduction='mean'):
        d_phi = self.d_phi

        # Output in (0,1), binary prob
        out_pos = torch.clamp(d_phi(x_labeled.detach()), 1e-6, 1 - 1e-6)
        out_unl = torch.clamp(d_phi(x_unlabeled.detach()), 1e-6, 1 - 1e-6)

        # Positive sample loss
        loss_pos_1 = F.binary_cross_entropy(out_pos, torch.ones_like(out_pos), reduction='none')
        loss_pos_0 = F.binary_cross_entropy(out_pos, torch.zeros_like(out_pos), reduction='none')
        # Unlabeled loss as negative
        loss_unl_0 = F.binary_cross_entropy(out_unl, torch.zeros_like(out_unl), reduction='none')

        if reduction == 'mean':
            R_p_pos = loss_pos_1.mean()
            R_p_neg = loss_pos_0.mean()
            R_u_neg = loss_unl_0.mean()
        elif reduction == 'sum':
            R_p_pos = loss_pos_1.sum()
            R_p_neg = loss_pos_0.sum()
            R_u_neg = loss_unl_0.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

        # PU risk: positive + max{0, unl_neg - pos_neg}
        pu_loss = mu * R_p_pos + torch.clamp(R_u_neg - mu * R_p_neg, min=0.0)
        return pu_loss
    

    # def pu_discriminator_loss(self, x_labeled, x_unlabeled, mu=0.5, reduction='mean', loss_func=(lambda x: torch.sigmoid(-x))):
    #     # 使用 d_phi 计算正样本和未标记样本的预测概率
    #     out_pos = torch.clamp(self.d_phi(x_labeled.detach()), 1e-6, 1 - 1e-6)  # 正样本的预测概率
    #     out_unl = torch.clamp(self.d_phi(x_unlabeled.detach()), 1e-6, 1 - 1e-6)  # 未标记样本的预测概率

    #     # 正样本损失（通过sigmoid计算概率）
    #     loss_pos = loss_func(out_pos)  # 计算正样本的概率
        
    #     # 反向正样本损失（通过sigmoid计算反向概率）
    #     loss_pos_inv = loss_func(-out_pos)  # 计算反向正样本的概率
        
    #     # 未标记样本损失（通过sigmoid计算负样本的概率）
    #     loss_unl = loss_func(-out_unl)  # 假设未标记样本作为负类，计算其损失

    #     # 对正样本和未标记样本进行加权
    #     if reduction == 'mean':
    #         loss_pos = loss_pos.mean()
    #         loss_pos_inv = loss_pos_inv.mean()
    #         loss_unl = loss_unl.mean()
    #     elif reduction == 'sum':
    #         loss_pos = loss_pos.sum()
    #         loss_pos_inv = loss_pos_inv.sum()
    #         loss_unl = loss_unl.sum()
    #     else:
    #         raise ValueError("reduction must be 'mean' or 'sum'")

    #     # 正负样本的损失加权
    #     positive_risk = mu * loss_pos  # 正样本损失
    #     negative_risk = -mu * loss_pos_inv + loss_unl  # 未标记样本损失

    #     # 确保负样本风险非负
    #     negative_risk = torch.clamp(negative_risk, min=0.0)  # 负样本损失不能为负值

    #     # 最终的PU损失
    #     pu_loss = positive_risk + negative_risk

    #     return pu_loss


    def loss_uncertainty_softmargin(self, energy_id, energy_ood, margin=0.0, tau=1,
                                squared=False, reduction='mean'):
        # 惩罚 ID 能量 > margin，以及 OOD 能量 < margin
        pos = F.softplus((energy_id - margin) / tau)       # 平滑替代 relu(energy_id - margin)
        neg = F.softplus((margin - energy_ood) / tau)      # 平滑替代 relu(margin - energy_ood)

        if squared:
            pos = pos ** 2
            neg = neg ** 2

        loss = pos + neg
        return loss.mean() if reduction == 'mean' else loss.sum()

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
    
    def select_topk_id_candidates(self, weights: torch.Tensor, mu: float, unlabeled_mask: torch.Tensor) -> torch.Tensor:
        """
        选择 unlabeled 中 OOD 权重最低的 Top‑mu 节点索引（即置信度最高的伪 ID）

        参数:
            weights: [N]，全图节点的 OOD 权重（越大越像 OOD）
            mu: 伪 ID 采样比例（保留 mu × N_unl 的伪 ID）
            unlabeled_mask: [N]，布尔张量，True 表示 unlabeled 节点

        返回:
            mask_cand: [N]，布尔张量，True 表示选中的伪 ID 节点
        """
        weights_unlabeled = weights[unlabeled_mask]
        N_unl = weights_unlabeled.size(0)
        k = int(mu * N_unl)

        if k == 0 or N_unl == 0:
            return torch.zeros_like(weights, dtype=torch.bool)

        # 从小到大排序，取权重最小的 k 个（越小越像 ID）
        sorted_weights, _ = torch.sort(weights_unlabeled, descending=False)
        thresh = sorted_weights[k - 1].item()

        mask_cand = torch.zeros_like(weights, dtype=torch.bool)
        mask_cand[unlabeled_mask] = weights_unlabeled <= thresh
        return mask_cand.view(-1)

    def detect_energy_head(self, e, data, T=1.0, cos_threshold=-1, prop_layers=2, alpha=0.5):
        """
        基于余弦相似度剪枝邻接后，再进行能量传播。
        cos_threshold: 保留余弦相似度大于该值的边
        """

        # 3. 传播
        # e = self.propagation(e, data.edge_index,
        #                      prop_layers=prop_layers, alpha=alpha)

        # 4. 拆分 ID 与 openset
        ind_idx, openset_idx = data.known_in_unseen_mask, data.unknown_in_unseen_mask
        neg_energy_ind = e[ind_idx]
        neg_energy_openset = e[openset_idx]
        return neg_energy_ind, neg_energy_openset
    
    def get_energy_score(self, logits, data, T=1.0, cos_threshold=-1, prop_layers=2, alpha=0.5):
        e = T * torch.logsumexp(logits / T, dim=-1)

        # 2. 基于余弦相似度剪枝邻接
        #    获取节点特征（这里用 logits 作 embedding）
        embeddings = logits  # [N, C]
        row, col = data.edge_index
        # 仅一阶边：逐边计算余弦相似度
        h_row = embeddings[row]  # [E, C]
        h_col = embeddings[col]  # [E, C]
        cos_sim = F.cosine_similarity(h_row, h_col, dim=-1)  # [E]
        mask = cos_sim >= cos_threshold
        # 构造剪枝后的 edge_index
        pruned_row = row[mask]
        pruned_col = col[mask]

        # 3. 传播
        e = self.propagation(e, (pruned_row, pruned_col),
                             prop_layers=prop_layers, alpha=alpha)
        return e
    
    def detect(self, logits, data, T=1.0, cos_threshold=-1, prop_layers=2, alpha=0.5):
        e = T * torch.logsumexp(logits / T, dim=-1)
        e = self.propagation(e, data.edge_index,
                             prop_layers=prop_layers, alpha=alpha)
        # 4. 拆分 ID 与 openset
        # ind_idx, openset_idx = data.train_mask, data.test_mask
        # neg_energy_ind = e[ind_idx]
        # neg_energy_openset = e[openset_idx]
        # return neg_energy_ind, neg_energy_openset
        return e

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
from torch_geometric.utils import homophily, to_undirected
import seaborn as sns

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
def compare_classwise_homophily_all(edge_index_clean, edge_index_attacked,
                                     y, train_mask, known_mask, unknown_mask,
                                     num_classes=None, figsize=(12, 4), save_path=None):
    """
    对比干净图与攻击图在每个类上的训练节点、known测试节点、unknown测试节点的同质性变化。
    """

    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    def per_node_homophily(edge_index, y):
        """
        返回每个节点的 homophily 分数: 它的邻居中标签相同的比例
        """
        edge_index = to_undirected(edge_index)
        row, col = edge_index
        same_label = (y[row] == y[col]).float()
        out = scatter(same_label, col, dim=0, dim_size=y.size(0), reduce='mean')
        return out  # shape [num_nodes]

    def get_classwise_hom(edge_index, mask, name):
        edge_index = to_undirected(edge_index)
        node_hom = per_node_homophily(edge_index, y)  # 每个节点的同质性
        class_hom = []
        for c in range(num_classes):
            class_mask = (y == c) & mask
            if class_mask.sum() > 0:
                avg_hom = node_hom[class_mask].mean().item()
            else:
                avg_hom = float('nan')
            class_hom.append(avg_hom)
        return class_hom

    def plot_lines(h_clean, h_attacked, label, ax):
        x = list(range(num_classes))
        ax.plot(x, h_clean, marker='o', label='Clean', color='tab:blue')
        ax.plot(x, h_attacked, marker='x', label='Attacked', color='tab:red')
        ax.set_title(f'{label} Node Homophily')
        ax.set_xlabel('Class')
        ax.set_ylabel('Avg Homophily')
        ax.set_xticks(x)
        ax.grid(True)
        ax.legend()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for mask, label, ax in zip(
        [train_mask, known_mask, unknown_mask],
        ['Train', 'Known in Test', 'Unknown in Test'],
        axes
    ):
        h_clean = get_classwise_hom(edge_index_clean, mask, label)
        h_attacked = get_classwise_hom(edge_index_attacked, mask, label)
        plot_lines(h_clean, h_attacked, label, ax)

    plt.tight_layout()
    if save_path:
        # plt.savefig(save_path, dpi=300)
        print(f'Saved figure to {save_path}')
    plt.show()

import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_scatter import scatter
import numpy as np

def compare_classwise_neighbor_label_dist(edge_index_clean, edge_index_attacked,
                                          y, train_mask, known_mask, unknown_mask,
                                          num_classes=None, figsize=(15, 5), save_path=None):
    """
    对比干净图与攻击图在每个类上的训练节点、known测试节点、unknown测试节点的
    一阶邻居标签分布变化（去掉自环）。
    """

    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    def per_node_neighbor_label_distribution(edge_index, y):
        """
        返回每个节点的邻居标签分布（形状: [num_nodes, num_classes]）
        """
        # 转无向 + 去掉自环
        edge_index, _ = remove_self_loops(to_undirected(edge_index))
        row, col = edge_index  # col 是被聚合的节点
        y_onehot = torch.nn.functional.one_hot(y[row], num_classes=num_classes).float()
        # 聚合邻居标签计数
        label_counts = scatter(y_onehot, col, dim=0, dim_size=y.size(0), reduce='sum')
        # 归一化为比例
        neighbor_dist = label_counts / label_counts.sum(dim=1, keepdim=True).clamp(min=1)
        return neighbor_dist  # [num_nodes, num_classes]

    def get_classwise_dist(edge_index, mask):
        """
        返回每个类在指定 mask 下的邻居标签分布（先按节点求均值，再按类求均值）
        """
        neighbor_dist = per_node_neighbor_label_distribution(edge_index, y)  # [N, C]
        class_dist = []
        for c in range(num_classes):
            class_mask = (y == c) & mask
            if class_mask.sum() > 0:
                avg_dist = neighbor_dist[class_mask].mean(dim=0)  # 每类的平均邻居标签分布
            else:
                avg_dist = torch.full((num_classes,), float('nan'))
            class_dist.append(avg_dist)
        return torch.stack(class_dist)  # [num_classes, num_classes]

    def plot_heatmap(dist_matrix, title, ax):
        im = ax.imshow(dist_matrix.cpu().numpy(), cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel("Neighbor Label")
        ax.set_ylabel("Node Class")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    for i, (mask, label) in enumerate([
        (train_mask, 'Train'),
        (known_mask, 'Known in Test'),
        (unknown_mask, 'Unknown in Test')
    ]):
        dist_clean = get_classwise_dist(edge_index_clean, mask)
        dist_attacked = get_classwise_dist(edge_index_attacked, mask)
        plot_heatmap(dist_clean, f'{label} - Clean', axes[i, 0])
        plot_heatmap(dist_attacked, f'{label} - Attacked', axes[i, 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    plt.show()

def compute_class_similarity_matrix(edge_index, y, num_classes, mask=None):
    """
    计算每个类之间的一阶邻居标签分布余弦相似度矩阵
    """
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    row, col = edge_index

    if mask is not None:
        node_mask = mask
    else:
        node_mask = torch.ones_like(y, dtype=torch.bool)

    # 计算每个节点的邻居标签分布直方图
    hist = torch.zeros((y.size(0), num_classes), device=y.device)
    for c in range(num_classes):
        indicator = (y[row] == c).float()
        hist[:, c] = scatter(indicator, col, dim=0, dim_size=y.size(0), reduce='sum')
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-12)  # 归一化

    # 类间相似度矩阵
    sim_matrix = torch.zeros((num_classes, num_classes), device=y.device)
    for m in range(num_classes):
        nodes_m = torch.where((y == m) & node_mask)[0]
        if len(nodes_m) == 0:
            continue
        for mp in range(num_classes):
            nodes_mp = torch.where((y == mp) & node_mask)[0]
            if len(nodes_mp) == 0:
                continue
            h_m = hist[nodes_m]
            h_mp = hist[nodes_mp]
            # 计算所有节点对的余弦相似度
            sim = torch.mm(h_m, h_mp.t()) / (
                h_m.norm(dim=1, keepdim=True) * h_mp.norm(dim=1).unsqueeze(0) + 1e-12
            )
            if m == mp:
                sim.fill_diagonal_(float('nan'))  # 类内相似度去掉自身
            sim_matrix[m, mp] = torch.nanmean(sim).item()
    return sim_matrix.cpu().numpy()

def compare_class_similarity(edge_index_clean, edge_index_attacked, y, num_classes, mask=None, save_path=None):
    sim_clean = compute_class_similarity_matrix(edge_index_clean, y, num_classes, mask)
    sim_attacked = compute_class_similarity_matrix(edge_index_attacked, y, num_classes, mask)
    diff = sim_attacked - sim_clean

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Clean Graph', 'Attacked Graph', 'Difference (Attacked - Clean)']
    data_list = [sim_clean, sim_attacked, diff]

    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, cmap='coolwarm', vmin=-1, vmax=1)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Class j")
        ax.set_ylabel("Class i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return sim_clean, sim_attacked, diff

def per_node_neighbor_label_distribution(edge_index, y, num_classes):
    """
    返回每个节点的邻居标签分布 [num_nodes, num_classes]
    """
    # 转无向 + 去掉自环
    edge_index, _ = remove_self_loops(to_undirected(edge_index))
    row, col = edge_index  # col 是被聚合的节点
    y_onehot = F.one_hot(y[row], num_classes=num_classes).float()
    # 聚合邻居标签计数
    label_counts = scatter(y_onehot, col, dim=0, dim_size=y.size(0), reduce='sum')
    # 归一化为比例
    neighbor_dist = label_counts / label_counts.sum(dim=1, keepdim=True).clamp(min=1)
    return neighbor_dist  # [num_nodes, num_classes]

def compute_classwise_distribution(edge_index, y, num_classes, mask):
    """
    返回每个类别在 mask 下的平均邻居标签分布
    """
    neighbor_dist = per_node_neighbor_label_distribution(edge_index, y, num_classes)  # [N, C]
    class_dist = []
    for c in range(num_classes):
        class_mask = (y == c) & mask
        if class_mask.sum() > 0:
            avg_dist = neighbor_dist[class_mask].mean(dim=0)  # 该类的平均邻居标签分布
        else:
            avg_dist = torch.full((num_classes,), float('nan'), device=y.device)
        class_dist.append(avg_dist)
    return torch.stack(class_dist)  # [C, C]

def compute_distribution_shift(edge_index, y, num_classes, train_mask, test_mask, eps=1e-12):
    dist_train = compute_classwise_distribution(edge_index, y, num_classes, train_mask)  # [C, C]
    dist_test  = compute_classwise_distribution(edge_index, y, num_classes, test_mask)   # [C, C]

    kl_per_class = []
    for c in range(num_classes):
        p = dist_train[c]
        q = dist_test[c]
        if torch.isnan(p).any() or torch.isnan(q).any():
            continue
        # 加 Laplace 平滑
        p = (p + eps) / (p.sum() + eps * num_classes)
        q = (q + eps) / (q.sum() + eps * num_classes)
        # 计算 KL(P || Q)
        kl = (p * torch.log(p / q)).sum().item()
        kl_per_class.append(kl)

    if len(kl_per_class) == 0:
        return float('nan')
    return sum(kl_per_class) / len(kl_per_class)


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss,self).__init__()

        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")

        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x)) / torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = 1
    
    def forward(self, inp, target, test=False):
        assert(inp.shape == target.shape)        

        if inp.is_cuda:
            self.prior = self.prior.cuda()

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        n_positive, n_unlabeled = torch.clamp(torch.sum(positive), min=self.min_count), torch.clamp(torch.sum(unlabeled), min=self.min_count)

        y_positive = self.loss_func(inp) * positive
        y_positive_inv = self.loss_func(-inp) * positive
        y_unlabeled = self.loss_func(-inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk

        return positive_risk + negative_risk

"""Torch module for GCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from grb.utils.normalize import GCNAdjNorm
import math
from moe_tools import GCNConv_moe
from torch.distributions import Laplace


class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 residual=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GCNConv_moe(n_features[i], n_features[i + 1], num_experts=4, noisy_gating=True, k=1))


        #self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    @property
    def model_name(self):
        return "gcndp"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    # def noise_layer(self, x):
    #     noise = torch.normal(0, 1, x.torch.Size)
    #     return x + noise

    def forward(self, x, adj):

        dp_del = 0.05;
        dp_eps = 1.0;
        noise_scale = 0.3 * math.sqrt(2 * math.log(1.25 / dp_del)) / dp_eps
        count = 0

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:

                x, expert_list = layer(x, adj)
                if count == 0:
                    gussian = torch.normal(0, 1, x.shape)
                    gussian = gussian.to('cuda:0')
                    x = x + noise_scale*gussian
                    # loc = torch.zeros(x.shape, dtype=torch.float)
                    # scale = torch.ones(x.shape, dtype=torch.float)
                    # noise = Laplace(loc, scale).sample()
                    # x = x + noise_scale * noise
                    count += 1

        return x


class GCNGC(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 residual=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0):
        super(GCNGC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features

        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GCNConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation,
                                       residual=residual,
                                       dropout=dropout))
        self.linear = nn.Linear(hidden_features[-1], out_features)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    @property
    def model_name(self):
        return "gcn"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, batch_index=None):

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj)
        if batch_index is not None:
            batch_size = int(torch.max(batch_index)) + 1
            out = torch.zeros(batch_size, x.shape[1]).to(x.device)
            out = out.scatter_add_(dim=0, index=batch_index.view(-1, 1).repeat(1, x.shape[1]), src=x)
        else:
            out = torch.sum(x, dim=0)
        out = self.dropout(self.linear(out))

        return out


class GCNConv(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 activation=None,
                 residual=False,
                 dropout=0.0):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        if residual:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None
        self.activation = activation

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, adj):

        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        if self.activation is not None:
            x = self.activation(x)
        if self.residual is not None:
            x = x + self.residual(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x

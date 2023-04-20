# Classes modified from
# https://github.com/lushleaf/varying-coefficient-net-with-functional-tr
# 03/9/2023 by Mauricio Tec (mauriciogtec@gmail.com)
# Simplified VCNet to VNetOutcome without the density model

import torch
import torch.nn as nn


class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        # x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'elu':
            self.act = nn.ELU()
        elif act == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def get_act_layer(s):
    if s == "relu":
        return nn.ReLU()
    elif s == "elu":
        return nn.ELU()
    elif s == "silu":
        return nn.SiLU()
    elif s == "tanh":
        return nn.Tanh()
    elif s == "sigmoid":
        return nn.Sigmoid()
    elif s == "id":
        return nn.Identity()
    else:
        raise NotImplementedError(s)


class CausalNet(nn.Module):
    def __init__(self, cfg_body, cfg_head, degree=2, knots=[0.33, 0.66], tmin=0.0, tmax=1.0, binary=False):
        super().__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        self.degree = degree
        self.knots = knots
        self.tmin = tmin
        self.tmax = tmax
        self.binary = binary


        # construct the body
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg_body):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                blocks.append(self.feature_weight)
            else:
                blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            blocks.append(get_act_layer(layer_cfg[3]))
        self.body = nn.Sequential(*blocks)

        # construct the dynamics network
        if not binary:
            blocks = []
            for layer_idx, layer_cfg in enumerate(cfg_head):
                if layer_idx == len(cfg_head)-1: # last layer
                    last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
                else:
                    blocks.append(
                        Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
            blocks.append(last_layer)
            self.Q = nn.Sequential(*blocks)
        else:
            heads = []
            for _ in range(2):
                blocks = []
                for layer_idx, layer_cfg in enumerate(cfg_head):
                    blocks.append(nn.Linear(layer_cfg[0], layer_cfg[1], bias=layer_cfg[2]))
                    blocks.append(get_act_layer(layer_cfg[3]))
                sub_head = nn.Sequential(*blocks)
                heads.append(sub_head)
            self.Q = nn.ModuleList(heads)

        
    def forward(self, t, x):
        hidden = self.body(x)
        if not self.binary:
            t = (t - self.tmin) / (self.tmax - self.tmin)
            t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
            Q = self.Q(t_hidden)
        else:
            Q = t * self.Q[1](hidden) + (1 - t) * self.Q[0](hidden)
        return Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

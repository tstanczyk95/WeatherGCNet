import torch.nn as nn
import torch
import math
from torch.autograd import Variable


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(unit_tcn, self).__init__()

        pad = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unit_gcn, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.B = nn.Parameter(torch.zeros(7, 7) + 1e-6)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = Variable(torch.eye(7), requires_grad=False)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

    def forward(self, x):
        N, C, T, V = x.size()

        f_in = x.contiguous().view(N, C * T, V)

        adj_mat = None
        self.A = self.A.cuda(x.get_device())
        adj_mat = self.B[:,:] + self.A[:,:]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)

        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)

        y = self.conv(torch.matmul(f_in, adj_mat_norm_d12).view(N, C, T, V))

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)    

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


    def forward(self, x):
        x = self.gcn1(x) + self.residual(x)
        x = self.tcn1(x)
        x = self.relu(x)

        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.data_bn = nn.BatchNorm1d(6 * 7) # channels (C) * vertices (V) ###

        self.l1 = TCN_GCN_unit(6, 16)
        self.l2 = TCN_GCN_unit(16, 32)
        self.l3 = TCN_GCN_unit(32, 64)

        self.conv_reduce_dim = unit_tcn(64, 4, kernel_size=1, stride=1) 

        self.fc = nn.Linear(840, 7) #  4 * 30 * 7 = 840

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 7))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
    
        x = self.conv_reduce_dim(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


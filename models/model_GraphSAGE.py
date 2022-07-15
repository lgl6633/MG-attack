import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
#from dgl.nn.pytorch.conv import SAGEConv
import scipy.sparse as sp
import random
import math
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class NewConvolution(nn.Module):
    """
    A Graph Convolution Layer for GraphSage
    """
    def __init__(self, in_features, out_features, bias=True):
        super(NewConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_1 = nn.Linear(in_features, out_features, bias=bias)
        self.W_2 = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W_1.weight.size(1))
        self.W_1.weight.data.uniform_(-stdv, stdv)
        self.W_2.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support_1 = self.W_1(input)
        support_2 = self.W_2(input)
        output = torch.mm(adj, support_2)
        output = output + support_1
        return output

class GraphSAGE(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSAGE, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)

        self.gc1 = NewConvolution(nfeat, nhid)
        self.gc2 = NewConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, adj, x, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque
from torch.distributions import Normal

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.logprobs_a = []
        self.logprobs_b = []
        self.rewards = []
        self.dists = []
        self.hists_a = []
        self.hists_b = []
        self.indices_a = []
        self.indices_b = []
    
    def clear(self):
        del self.states[:]
        del self.logprobs_a[:]
        del self.logprobs_b[:]
        del self.rewards[:]
        del self.dists[:]
        del self.hists_a[:]
        del self.hists_b[:]
        del self.indices_a[:]
        del self.indices_b[:]

class VIPActorIPD(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_layers=1):
        super(VIPActorIPD, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.hidden_size = hidden_size

        self.gru = nn.GRU(in_size, hidden_size, num_layers, batch_first=True)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        x = F.relu(self.hidden(x))
        return output, F.softmax(self.linear(x).flatten(), dim=0)
    
    def batch_forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]), h_0)
        else:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]))
        x = F.relu(self.hidden(x))
        return output, F.softmax(self.linear(x), dim=2)
    
class VIPCriticIPD(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1, gru=None):
        super(VIPCriticIPD, self).__init__()

        self.in_size = in_size
        self.device = device
        if gru:
            self.gru = gru
        else:
            self.gru = nn.GRU(in_size, hidden_size, num_layers, batch_first=True)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        x = F.relu(self.hidden(x))
        return output, self.linear(F.relu(x)).flatten()
    
    def batch_forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]), h_0)
        else:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]))
        x = F.relu(self.hidden(x))
        return output, self.linear(F.relu(x))

class VIPActor(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_layers=1):
        super(VIPActor, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.first = nn.Linear(self.in_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.hidden = nn.Linear(self.hidden_size, self.in_size)
        self.linear = nn.Linear(self.in_size, self.out_size)

    def forward(self, x, h_0=None):
        inpt = x
        x = F.relu(self.first(x))
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        x = F.relu(self.hidden(x + inpt))
        return output, F.softmax(self.linear(x).flatten(), dim=0)

    def batch_forward(self, x, pi_b=None, h_0=None):
        inpt = x
        x = F.relu(self.first(x))
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]), h_0)
        else:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]))
        x = F.relu(self.hidden(x + inpt))
        return output, F.softmax(self.linear(x), dim=2)

class HistoryAggregator(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_layers=1):
        super(HistoryAggregator, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(in_size, out_size, num_layers, batch_first = True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, hidden = self.lstm(x)
        return F.relu(x[:,-1,:])
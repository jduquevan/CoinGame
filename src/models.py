import math
import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import init, Parameter

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

# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.device = device
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight).to(self.device),
                        self.bias + self.sigma_bias * Variable(self.epsilon_bias).to(self.device))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class VIPActorIPD(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_layers=1, noisy=False):
        super(VIPActorIPD, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.hidden_size = hidden_size

        self.gru = nn.GRU(in_size, hidden_size, num_layers, batch_first=True)
        if noisy:
            self.hidden = NoisyLinear(hidden_size, hidden_size, device)
            self.linear = NoisyLinear(hidden_size, out_size, device)
        else:
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
    
    def sample_noise(self):
        self.hidden.sample_noise()
        self.linear.sample_noise()
    
class VIPCriticIPD(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1, gru=None, noisy=False):
        super(VIPCriticIPD, self).__init__()

        self.in_size = in_size
        self.device = device
        if gru:
            self.gru = gru
        else:
            self.gru = nn.GRU(in_size, hidden_size, num_layers, batch_first=True)

        if noisy:
            self.hidden = NoisyLinear(hidden_size, hidden_size, device)
            self.linear = NoisyLinear(hidden_size, 1, device)
        else:
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
    
    def sample_noise(self):
        self.hidden.sample_noise()
        self.linear.sample_noise()

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
        x = F.relu(self.hidden(x + F.relu(self.first(inpt))))
        return output, F.softmax(self.linear(x).flatten(), dim=0)

    def batch_forward(self, x, pi_b=None, h_0=None):
        inpt = x
        x = F.relu(self.first(x))
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]), h_0)
        else:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]))
        x = F.relu(self.hidden(x + F.relu(self.first(inpt))))
        return output, F.softmax(self.linear(x), dim=2)

class VIPCritic(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1):
        super(VIPCritic, self).__init__()

        self.in_size = in_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.first = nn.Linear(self.in_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.hidden = nn.Linear(self.hidden_size, self.in_size)
        self.linear = nn.Linear(self.in_size, 1)

    def forward(self, x, h_0=None):
        inpt = x
        x = F.relu(self.first(x))
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        x = F.relu(self.hidden(x + F.relu(self.first(inpt))))
        return output, self.linear(x).flatten()

    def batch_forward(self, x, pi_b=None, h_0=None):
        inpt = x
        x = F.relu(self.first(x))
        self.gru.flatten_parameters()
        if h_0 is not None:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]), h_0)
        else:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]))
        x = F.relu(self.hidden(x + F.relu(self.first(inpt))))
        return output, self.linear(x)

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
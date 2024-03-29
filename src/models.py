import math
import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from collections import namedtuple, deque
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import init, Parameter

Transition = namedtuple('Transition',
                        ('states_a', 
                         'rewards_a',
                         'rewards_b',
                         'log_probs_a',
                         'log_probs_b',
                         'hiddens_a',
                         'values_b',
                         'causal_rewards_b'))

class RolloutBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AbstractNoisyLayer(nn.Module, ABC):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float,
    ):
        super().__init__()

        self.sigma = sigma
        self.input_features = input_features
        self.output_features = output_features

        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))

        self.register_buffer('epsilon_input', torch.FloatTensor(input_features))
        self.register_buffer('epsilon_output', torch.FloatTensor(output_features))

    def forward(
            self,
            x: torch.Tensor,
            sample_noise: bool = True
    ) -> torch.Tensor:
        if not self.training:
            return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        return nn.functional.linear(x, weight=self.weight, bias=self.bias)

    @property
    def weight(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bias(self) -> torch.Tensor:
        raise NotImplementedError

    def sample_noise(self) -> None:
        raise NotImplementedError

    def parameter_initialization(self) -> None:
        raise NotImplementedError

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        noise = torch.FloatTensor(features).uniform_(-self.bound, self.bound).to(self.mu_bias.device)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


class IndependentNoisyLayer(AbstractNoisyLayer):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float = 0.017,
    ):
        super().__init__(
            input_features=input_features,
            output_features=output_features,
            sigma=sigma
        )

        self.bound = (3 / input_features) ** 0.5
        self.parameter_initialization()
        self.sample_noise()

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * self.epsilon_weight[0] + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_bias + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_bias = self.get_noise_tensor((self.output_features,))
        self.epsilon_weight = self.get_noise_tensor((self.output_features, self.input_features))

    def parameter_initialization(self) -> None:
        self.sigma_bias.data.fill_(self.sigma)
        self.sigma_weight.data.fill_(self.sigma)
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)


class FactorisedNoisyLayer(AbstractNoisyLayer):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float = 0.5,
    ):
        super().__init__(
            input_features=input_features,
            output_features=output_features,
            sigma=sigma
        )

        self.bound = input_features**(-0.5)
        self.parameter_initialization()
        self.sample_noise()

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_output + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def parameter_initialization(self) -> None:
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma * self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.sigma_weight.data.fill_(self.sigma * self.bound)

# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, sigma_init=0.1, bias=True):
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

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        if noisy:
            self.first = FactorisedNoisyLayer(in_size, hidden_size)
            self.hidden = FactorisedNoisyLayer(hidden_size, hidden_size)
            self.linear = FactorisedNoisyLayer(hidden_size, out_size)
        else:
            self.first = nn.Linear(in_size, hidden_size)
            self.hidden = nn.Linear(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        x = F.relu(self.first(x))
        x = F.relu(self.hidden(x))
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        # x = F.relu(self.hidden(x))
        return output, F.softmax(self.linear(x).flatten(), dim=0)
    
    def batch_forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        x = F.relu(self.first(x))
        x = F.relu(self.hidden(x))
        if h_0 is not None:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]), h_0)
        else:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]))
        # x = F.relu(self.hidden(x))
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
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        if noisy:
            self.first = FactorisedNoisyLayer(in_size, hidden_size)
            self.hidden = FactorisedNoisyLayer(hidden_size, hidden_size)
            self.linear = FactorisedNoisyLayer(hidden_size, 1)
        else:
            self.first = nn.Linear(in_size, hidden_size)
            self.hidden = nn.Linear(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        x = F.relu(self.first(x))
        x = F.relu(self.hidden(x))
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        # x = F.relu(self.hidden(x))
        return output, self.linear(F.relu(x)).flatten()
    
    def batch_forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        x = F.relu(self.first(x))
        x = F.relu(self.hidden(x))
        if h_0 is not None:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]), h_0)
        else:
            output, x = self.gru(x.reshape(x.shape[0], 1, x.shape[1]))
        # x = F.relu(self.hidden(x))
        return output, self.linear(F.relu(x))
    
    def sample_noise(self):
        self.hidden.sample_noise()
        self.linear.sample_noise()

class VIPActor(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_layers=1, noisy=False):
        super(VIPActor, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        if noisy:
            self.first = FactorisedNoisyLayer(in_size, hidden_size)
            self.hidden = FactorisedNoisyLayer(hidden_size, hidden_size)
            self.linear = FactorisedNoisyLayer(hidden_size, out_size)
        else:
            self.first = nn.Linear(self.in_size, self.hidden_size)
            self.hidden = nn.Linear(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, out_size)

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
    
    def sample_noise(self):
        self.first.sample_noise()
        self.hidden.sample_noise()
        self.linear.sample_noise()

class VIPCritic(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1, noisy=False):
        super(VIPCritic, self).__init__()

        self.in_size = in_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        if noisy:
            self.first = FactorisedNoisyLayer(in_size, hidden_size)
            self.hidden = FactorisedNoisyLayer(hidden_size, hidden_size)
            self.linear = FactorisedNoisyLayer(hidden_size, 1)
        else:
            self.first = nn.Linear(self.in_size, self.hidden_size)
            self.hidden = nn.Linear(hidden_size, hidden_size)
            self.linear = nn.Linear(hidden_size, 1)

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
    
    def sample_noise(self):
        self.first.sample_noise()
        self.hidden.sample_noise()
        self.linear.sample_noise()

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
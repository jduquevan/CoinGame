import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import reduce

from .models import VIPActor, VIPActorIPD, VIPCriticIPD, HistoryAggregator
from .optimizers import ExtraAdam, OptimisticAdam
from .utils import magic_box

class BaseAgent():
    def __init__(self,
                 gamma,
                 opt_type,
                 critic_opt_type,
                 exp_opt_type,
                 other_opt_type,
                 device,
                 n_actions,
                 obs_shape):
        self.steps_done = 0
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.opt_type = opt_type
        self.critic_opt_type = critic_opt_type
        self.exp_opt_type = exp_opt_type
        self.other_opt_type = other_opt_type
        
        self.obs_size = reduce(lambda a, b: a * b, self.obs_shape)

class AlwaysCooperateAgent():
    def __init__(self, is_p_1, device, n_actions):
        self.cum_steps = 0
        self.device = device
        self.n_actions = n_actions
        self.is_p_1 = torch.tensor(is_p_1).to(self.device)

    def select_action(self, env):
        action = env.get_coop_action(self.is_p_1)
        dist = torch.nn.functional.one_hot(action, self.n_actions).to(self.device)
        return action, dist


class AlwaysDefectAgent():
    def __init__(self, is_p_1, device, n_actions):
        self.cum_steps = 0
        self.device = device
        self.n_actions = n_actions
        self.is_p_1 = torch.tensor(is_p_1).to(self.device)

    def select_action(self, env):
        action = env.get_moves_shortest_path_to_coin(self.is_p_1)
        dist = torch.nn.functional.one_hot(action, self.n_actions).to(self.device)
        return action, dist

class VIPAgent(BaseAgent):
    def __init__(self,
                 config,
                 optim_config,
                 critic_optim_config,
                 batch_size,
                 rollout_len,
                 hidden_size,
                 entropy_weight,
                 inf_weight,
                 device,
                 n_actions,
                 obs_shape,
                 is_cg):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.cum_steps = 0
        self.batch_size = batch_size
        self.rollout_len = rollout_len
        self.hidden_size = hidden_size
        self.entropy_weight = entropy_weight
        self.inf_weight = inf_weight
        self.n_actions = n_actions
        self.is_cg = is_cg
        self.transition: list = list()

        if self.is_cg:
            self.in_size = self.obs_size + 2*self.n_actions
        else:
            self.in_size = 2*self.n_actions

        self.actor = VIPActorIPD(in_size=self.in_size,
                                 out_size=self.n_actions,
                                 device=self.device,
                                 hidden_size=self.hidden_size)
        self.critic = VIPCriticIPD(in_size=self.in_size,
                                   device=self.device,
                                   hidden_size=self.hidden_size)
        self.target = VIPCriticIPD(in_size=self.in_size,
                                   device=self.device,
                                   hidden_size=self.hidden_size)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target.to(self.device)
        
        if self.opt_type.lower() == "sgd":
            self.optimizer = optim.SGD(list(self.actor.parameters()), 
                                       lr=optim_config["lr"],
                                       momentum=optim_config["momentum"],
                                       weight_decay=optim_config["weight_decay"],
                                       maximize=True)
        elif self.opt_type.lower() == "adam":
            self.optimizer = optim.Adam(list(self.actor.parameters()), 
                                        lr=optim_config["lr"],
                                        weight_decay=optim_config["weight_decay"],
                                        maximize=True)
        elif self.opt_type.lower() == "eg":
            self.optimizer = ExtraAdam(list(self.actor.parameters()),
                                       lr=optim_config["lr"],
                                       betas=(optim_config["beta_1"], optim_config["beta_2"]),
                                       weight_decay=optim_config["weight_decay"])
        elif self.opt_type.lower() == "om":
            self.optimizer = OptimisticAdam(list(self.actor.parameters()),
                                            lr=optim_config["lr"],
                                            betas=(optim_config["beta_1"], optim_config["beta_2"]),
                                            weight_decay=optim_config["weight_decay"])
        
        if self.critic_opt_type.lower() == "sgd":
            self.critic_optimizer = optim.SGD(list(self.critic.parameters()), 
                                              lr=critic_optim_config["lr"],
                                              momentum=critic_optim_config["momentum"],
                                              weight_decay=critic_optim_config["weight_decay"])
        elif self.critic_opt_type.lower() == "adam":
            self.critic_optimizer = optim.Adam(list(self.critic.parameters()), 
                                               lr=critic_optim_config["lr"],
                                               weight_decay=critic_optim_config["weight_decay"])
        elif self.critic_opt_type.lower() == "eg":
            self.critic_optimizer = ExtraAdam(list(self.critic.parameters()),
                                              lr=critic_optim_config["lr"],
                                              betas=(critic_optim_config["beta_1"], critic_optim_config["beta_2"]),
                                              weight_decay=critic_optim_config["weight_decay"])
        elif self.critic_opt_type.lower() == "om":
            self.critic_optimizer = OptimisticAdam(list(self.critic.parameters()),
                                                   lr=critic_optim_config["lr"],
                                                   betas=(critic_optim_config["beta_1"], critic_optim_config["beta_2"]),
                                                   weight_decay=critic_optim_config["weight_decay"])
            
    def eval(self):
        self.critic.eval()
        self.actor.eval()

    def train(self):
        self.critic.train()
        self.actor.train()
            
    def compute_value_loss(self, values, targets, rewards):
        values = torch.permute(torch.stack(values).reshape(-1, self.batch_size), (1, 0))[:, 0:-1]
        targets = torch.permute(torch.stack(targets).reshape(-1, self.batch_size), (1, 0))[:, 1:]
        rewards = torch.permute(torch.stack(rewards).reshape(self.rollout_len, -1), (1, 0))[:, 0:-1]

        est_values = rewards + self.gamma * targets

        value_loss = (values - est_values).flatten().norm(dim=0, p=2)
        return value_loss
    
    def compute_reinforce_loss(self, log_probs_a, log_probs_b, states_a, rewards_a, rewards_b, hiddens_a, values_b, causal_rewards_b, is_cg):
        states_a = torch.permute(torch.stack(states_a), (1, 0, 2))
        rewards_a = torch.permute(torch.stack(rewards_a).reshape(self.rollout_len, -1), (1, 0))
        rewards_b = torch.permute(torch.stack(rewards_b).reshape(self.rollout_len, -1), (1, 0))
        log_probs_a = torch.permute(torch.stack(log_probs_a).reshape(self.rollout_len, -1), (1, 0))
        log_probs_b = torch.permute(torch.stack(log_probs_b).reshape(self.rollout_len, -1), (1, 0))
        hiddens_a = torch.permute(torch.stack(hiddens_a).reshape(self.rollout_len, self.batch_size, -1)[0:-1, :, :], (1, 0, 2))
        values_b = torch.permute(torch.stack(values_b).reshape(self.rollout_len, self.batch_size), (1, 0))

        gammas = torch.tensor(self.gamma).repeat(self.batch_size, self.rollout_len - 1).to(self.device)
        gammas = torch.exp(torch.cumsum(torch.log(gammas), dim=1))
        gammas = torch.cat([torch.ones(self.batch_size, 1).to(self.device), gammas], dim=1)
        
        # import pdb; pdb.set_trace()
        if is_cg:
            h_s, values_no_hidden_a = self.target.batch_forward(states_a.reshape(-1, self.obs_size + self.n_actions*2)[0:self.batch_size, :])
            h_t, values_hidden_a = self.target.batch_forward(states_a.reshape(-1, self.obs_size + self.n_actions*2)[self.batch_size:, :], 
                                                            hiddens_a.reshape(1, -1, self.actor.hidden_size))
        else:
            h_s, values_no_hidden_a = self.target.batch_forward(states_a.reshape(-1, self.n_actions*2)[0:self.batch_size, :])
            h_t, values_hidden_a = self.target.batch_forward(states_a.reshape(-1, self.n_actions*2)[self.batch_size:, :], 
                                                            hiddens_a.reshape(1, -1, self.actor.hidden_size))
        values_a = torch.cat([values_no_hidden_a, values_hidden_a], dim=1).reshape(self.batch_size, self.rollout_len).detach()
        curr_state_vals_a = values_a[:, 0:-1]
        next_state_vals_a = values_a[:, 1:]

        advantages = rewards_a[:, 0:-1] + (self.gamma*next_state_vals_a - curr_state_vals_a).detach()
        advantages = (advantages - torch.mean(advantages))/torch.std(advantages)

        future_returns_b = []
        for i in range(self.rollout_len):
            if i>0:
                future_returns_b.append(torch.sum(causal_rewards_b[i][:, i:] * gammas[:, :-i], dim=1))
            else:
                future_returns_b.append(torch.sum(causal_rewards_b[i][:, i:] * gammas, dim=1))

        future_returns_b = torch.permute(torch.stack(future_returns_b), (1, 0)) - values_b.detach()
        future_returns_b = (future_returns_b - torch.mean(future_returns_b))/torch.std(future_returns_b)

        positive_adv_ratio = torch.sum(advantages>0)/(advantages.shape[0]*advantages.shape[1])
        positive_ret_ratio = torch.sum(future_returns_b>0)/(future_returns_b.shape[0]*future_returns_b.shape[1])

        # mask= torch.logical_or((advantages<0)*(future_returns_b[:, 1:]>0), (advantages>0)*(future_returns_b[:, 1:]<0))
        mask=(advantages<0)*(future_returns_b[:, 1:]>0)
        # mask=1*(advantages<0)*(future_returns_b[:, 1:]>0) - 1*(advantages<0)*(future_returns_b[:, 1:]<0)
        # mask= torch.torch.logical_not((advantages<0)*(future_returns_b[:, 1:]<0))
        # mask = torch.ones(self.batch_size, self.rollout_len-1).to(self.device) -1*(advantages<0)*(future_returns_b[:, 1:]<0)

        pg_loss = torch.mean(torch.sum(log_probs_a[:, 0:-1] * advantages * gammas[:, 0:-1], dim=1))
        inf_loss = torch.mean(torch.sum(future_returns_b[:, 0:-1] * advantages * mask, dim=1))

        # future_log_probs_a = torch.flip(torch.cumsum(torch.flip(log_probs_a, [1]), 1), [1])
        # mask_ex = torch.torch.logical_not((advantages<0)*(returns_b[:, 1:]<0))
        # mask_neg = -1*(advantages<0)*(returns_b[:, 1:]>0)
        # mask = torch.logical_or(mask, (advantages>0)*(returns_b[:, 1:]<0))

        
        # inf_loss = torch.mean(torch.sum(future_log_probs_a[:, 0:-1] * returns_b[:, 0:-1] * advantages * mask, dim=1))

        return pg_loss - self.inf_weight * inf_loss, positive_adv_ratio, positive_ret_ratio
    
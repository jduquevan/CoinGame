import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from functools import reduce

from .models import VIPActor, HistoryAggregator, RolloutBuffer
from .optimizers import ExtraAdam

class BaseAgent():
    def __init__(self,
                 gamma,
                 opt_type,
                 device,
                 n_actions,
                 obs_shape):
        self.steps_done = 0
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.opt_type = opt_type
        
        self.obs_size = reduce(lambda a, b: a * b, self.obs_shape)

class VIPAgent(BaseAgent):
    def __init__(self,
                 config,
                 optim_config,
                 batch_size,
                 rollout_len,
                 representation_size,
                 hidden_size,
                 history_len,
                 collab_weight,
                 exploit_weight,
                 entropy_weight,
                 device,
                 n_actions,
                 obs_shape,
                 model,
                 action_model):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.cum_steps = 0
        self.batch_size = batch_size
        self.rollout_len = rollout_len
        self.representation_size =representation_size
        self.hidden_size = hidden_size
        self.history_len = history_len
        self.exploit_weight = exploit_weight
        self.collab_weight = collab_weight
        self.entropy_weight = entropy_weight
        self.n_actions = n_actions
        self.transition: list = list()

        self.qa_module = HistoryAggregator(in_size=self.obs_size + 2*n_actions,
                                                    out_size=self.representation_size,
                                                    device=self.device,
                                                    hidden_size=self.hidden_size)
        
        self.actor = VIPActor(in_size=self.obs_size + 2*n_actions + self.representation_size,
                              out_size=self.n_actions,
                              device=self.device,
                              hidden_size=self.hidden_size)
        self.qa_module.to(self.device)
        self.actor.to(self.device)
        self.model = model
        self.action_model = action_model

        if self.opt_type.lower() == "sgd":
            self.optimizer = optim.SGD(list(self.actor.parameters()) + 
                                       list(self.qa_module.parameters()), 
                                       lr=optim_config["lr"],
                                       momentum=optim_config["momentum"],
                                       weight_decay=optim_config["weight_decay"],
                                       maximize=True)
        elif self.opt_type.lower() == "adam":
            self.optimizer = optim.Adam(list(self.actor.parameters()) + 
                                        list(self.qa_module.parameters()), 
                                        lr=optim_config["lr"],
                                        weight_decay=optim_config["weight_decay"],
                                        maximize=True)
        elif self.opt_type.lower() == "eg":
            self.optimizer = ExtraAdam(list(self.actor.parameters()) + 
                                       list(self.qa_module.parameters()),
                                       lr=optim_config["lr"],
                                       betas=(optim_config["beta_1"], optim_config["beta_2"]),
                                       weight_decay=optim_config["weight_decay"])
        
        self.kl_optimizer = optim.Adam(list(self.actor.parameters()) + 
                                       list(self.qa_module.parameters()), 
                                       lr=optim_config["lr"],
                                       weight_decay=optim_config["weight_decay"],
                                       maximize=False)

    def get_agent_representation(self, state_a, state_b, agent, h_a, h_b):
        no_info = -1 * torch.ones(self.representation_size).to(self.device)
        qa_hist = []

        for i in range(self.history_len):
            h_a, dist_a = self.actor(torch.cat([state_a, no_info]), h_a)
            h_b, dist_b = agent.actor(torch.cat([state_b, no_info]), h_b)
            action_a = torch.argmax(dist_a)
            action_b = torch.argmax(dist_b)
            
            last_a = torch.zeros(self.n_actions).to(self.device)
            last_b = torch.zeros(self.n_actions).to(self.device)
            last_a = last_a.scatter(0, action_a, 1)
            last_b = last_b.scatter(0, action_b, 1)
            
            obs, r, _, _ = self.action_model.step([action_a, action_b])
            obs_a, obs_b = obs

            state_a = torch.cat([obs_a.flatten(), last_a, last_b])
            state_b = torch.cat([obs_b.flatten(), last_b, last_a])
            
            qa_hist.append(torch.cat([obs_a.flatten(), dist_a.detach(), dist_b]))

        qa_hist = torch.stack(qa_hist).reshape(1, self.history_len, -1)
        agent_r = self.qa_module(qa_hist)
        return agent_r
    
    def get_agent_representations(self, states_a, states_b, agent, h_a, h_b):
        no_info = -1 * torch.ones(self.batch_size, self.representation_size).to(self.device)
        qa_hist = []

        for i in range(self.history_len):
            h_a, dist_a = self.actor.batch_forward(torch.cat([states_a, no_info], dim=1), h_a)
            h_b, dist_b = agent.actor.batch_forward(torch.cat([states_b, no_info], dim=1), h_b)

            h_a = torch.permute(h_a, (1, 0, 2))
            h_b = torch.permute(h_b, (1, 0, 2))
            dist_a = dist_a.reshape(self.batch_size, -1)
            dist_b = dist_b.reshape(self.batch_size, -1)

            actions_a = torch.argmax(dist_a, dim=1)
            actions_b = torch.argmax(dist_b, dim=1)
            
            last_a = torch.nn.functional.one_hot(actions_a, self.n_actions)
            last_b = torch.nn.functional.one_hot(actions_b, self.n_actions)
            
            obs, r, _, _ = self.model.step([actions_a, actions_b])
            obs_a, obs_b = obs

            states_a = torch.cat([obs_a.reshape(self.batch_size, -1), last_a, last_b], dim=1)
            states_b = torch.cat([obs_b.reshape(self.batch_size, -1), last_b, last_a], dim=1)
            
            qa_hist.append(torch.cat([obs_a.reshape(self.batch_size, -1), dist_a.detach(), dist_b], dim=1))

        qa_hist = torch.permute(torch.stack(qa_hist), (1, 0, 2))
        agent_r = self.qa_module(qa_hist)
        return agent_r
    
    def select_action(self, state_a, state_b, agent, h_a, h_b):
        self.steps_done += 1
        agent_r = self.get_agent_representation(state_a, state_b, agent, h_a, h_b)
        h_a, dist_a = self.actor(torch.cat([state_a, agent_r.flatten()]), h_a)
        action = torch.tensor([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())],
                              requires_grad=False,
                              device=self.device)
        return h_a, action
    
    def compute_kl_divergence(self, state_a, state_b, agent):
        _, _, h_a, h_b, _, _ = self.transition
        no_info = -1 * torch.ones(self.representation_size).to(self.device)
        agent_r = self.get_agent_representation(state_a, state_b, agent, h_a, h_b)
        _, pi_info = self.actor(torch.cat([state_a, agent_r.flatten()]), h_a)
        _, pi_no_info = self.actor(torch.cat([state_a, no_info]), h_a)
        kl = torch.sum(pi_no_info * torch.log(torch.div(pi_no_info, pi_info)))
        return kl

    def compute_pg_loss(self, agent, agent_t=1):
        self.cum_steps = self.cum_steps + 1
        steps = self.cum_steps

        # Parallel Monte-carlo rollouts
        t_rewards = []
        log_probs_a = []
        log_probs_b = []
        obs_a, obs_b, h_a, h_b, last_a, last_b = self.transition

        obs_a = obs_a.repeat((self.batch_size, 1, 1, 1)).reshape((self.batch_size,-1))
        obs_b = obs_b.repeat((self.batch_size, 1, 1, 1)).reshape((self.batch_size,-1))

        last_a = last_a.repeat((self.batch_size, 1, 1, 1)).reshape((self.batch_size,-1))
        last_b = last_b.repeat((self.batch_size, 1, 1, 1)).reshape((self.batch_size,-1))

        h_a = h_a.repeat((self.batch_size, 1, 1, 1)).reshape((self.batch_size, 1,-1))
        h_a = torch.permute(h_a, (1, 0, 2))

        h_b = h_b.repeat((self.batch_size, 1, 1, 1)).reshape((self.batch_size, 1,-1))
        h_b = torch.permute(h_b, (1, 0, 2))

        for i in range(self.rollout_len):
            states_a = torch.cat([obs_a, last_a, last_b], dim=1)
            states_b = torch.cat([obs_b, last_b, last_a], dim=1)

            if steps % self.rollout_len == 0:
                h_a, h_b = None, None
                last_a = -1*torch.ones(self.batch_size, self.n_actions).to(self.device)
                last_b = -1*torch.ones(self.batch_size, self.n_actions).to(self.device)

            reps_a = self.get_agent_representations(states_a, states_b, agent, h_a, h_b)
            reps_b = self.get_agent_representations(states_b, states_a, self, h_b, h_a)

            h_a, dists_a = self.actor.batch_forward(torch.cat([states_a, reps_a], dim=1), h_a)
            h_b, dists_b = agent.actor.batch_forward(torch.cat([states_b, reps_b], dim=1), h_b)

            h_a = torch.permute(h_a, (1, 0, 2))
            h_b = torch.permute(h_b, (1, 0, 2))
            dists_a = dists_a.reshape((self.batch_size, -1))
            dists_b = dists_b.reshape((self.batch_size, -1))
            
            actions_a = torch.tensor([np.random.choice(self.n_actions, p=dist.cpu().detach().numpy()) for dist in dists_a],
                                requires_grad=False,
                                device=self.device)
            actions_b = torch.tensor([np.random.choice(self.n_actions, p=dist.cpu().detach().numpy()) for dist in dists_b],
                                requires_grad=False,
                                device=self.device)

            last_a = torch.nn.functional.one_hot(actions_a, self.n_actions)
            last_b = torch.nn.functional.one_hot(actions_b, self.n_actions)
            
            a_t_probs = dists_a.gather(1, actions_a.reshape(-1, 1)).reshape(self.batch_size)
            b_t_probs = dists_b.gather(1, actions_b.reshape(-1, 1)).reshape(self.batch_size)

            log_probs_a.append(torch.log(a_t_probs))
            log_probs_b.append(torch.log(b_t_probs))

            if agent_t == 1:
                obs, r, _, _ = self.model.step([actions_a, actions_b])
                obs_a, obs_b = obs
                r1, r2 = r
            else:
                obs, r, _, _ = self.model.step([actions_b, actions_a])
                obs_b, obs_a = obs
                r2, r1 = r

            obs_a = obs_a.reshape((self.batch_size, -1))
            obs_b = obs_b.reshape((self.batch_size, -1))
            
            t_rewards.append(r1)
            
            steps = steps + 1
        gammas = torch.tensor(self.gamma).repeat(self.batch_size, self.rollout_len - 1).to(self.device)
        gammas = torch.exp(torch.cumsum(torch.log(gammas), dim=1))
        gammas = torch.cat([torch.ones(self.batch_size, 1).to(self.device), gammas], dim=1)
        rewards_t = torch.permute(torch.stack(t_rewards), (1, 0))
        returns = torch.sum(rewards_t*gammas, dim=1)
        log_probs = torch.permute(torch.stack(log_probs_a) + torch.stack(log_probs_b), (1, 0))
        sum_log_probs = torch.sum(log_probs, dim=1)
        pg_loss = torch.mean(returns * sum_log_probs)

        return pg_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import reduce

from .models import VIPActor, HistoryAggregator, RolloutBuffer
from .optimizers import ExtraAdam, OptimisticAdam
from .utils import magic_box

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
                 batch_size,
                 rollout_len,
                 representation_size,
                 hidden_size,
                 history_len,
                 collab_weight,
                 exploit_weight,
                 entropy_weight,
                 rep_dropout,
                 device,
                 n_actions,
                 obs_shape,
                 model,
                 action_models,
                 action_model=None,
                 qa_module=None):
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
        self.rep_dropout = rep_dropout
        self.n_actions = n_actions
        self.transition: list = list()
        
        if qa_module != None:
            self.qa_module = qa_module
        else:
            self.qa_module = HistoryAggregator(in_size=self.obs_size + n_actions,
                                               out_size=self.representation_size,
                                               device=self.device,
                                               hidden_size=self.hidden_size)
        
        self.actor = VIPActor(in_size=self.obs_size + self.representation_size,
                              out_size=self.n_actions,
                              device=self.device,
                              hidden_size=self.hidden_size)
        
        self.qa_module.to(self.device)
        self.actor.to(self.device)
        self.model = model
        self.action_model = action_model
        self.action_models = action_models

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
        elif self.opt_type.lower() == "om":
            self.optimizer = OptimisticAdam(list(self.actor.parameters()) + 
                                            list(self.qa_module.parameters()),
                                            lr=optim_config["lr"],
                                            betas=(optim_config["beta_1"], optim_config["beta_2"]),
                                            weight_decay=optim_config["weight_decay"])

    def eval(self):
        self.qa_module.eval()
        self.actor.eval()
    
    def train(self):
        self.qa_module.train()
        self.actor.train()

    def get_fixed_representation(self, state_a, agent, h_a, env):
        no_info = torch.zeros(self.representation_size).to(self.device)
        qa_hist = []
        for i in range(self.history_len):
            h_a, dist_a = self.actor(torch.cat([state_a, no_info]), h_a)
            action_b, dist_b = agent.select_action(env)
            action_a = torch.multinomial(dist_a, 1)
            
            obs, r, _, _ = self.action_model.step([action_a, action_b])
            obs_a, obs_b = obs
            
            qa_hist.append(torch.cat([obs_a.flatten(), dist_b.flatten()]))

        qa_hist = torch.stack(qa_hist).reshape(1, self.history_len, -1)
        agent_r = self.qa_module(qa_hist)
        return agent_r

    def get_fixed_representations(self, states_a, agent, h_a, env):
        no_info = torch.zeros(self.batch_size, self.representation_size).to(self.device)
        qa_hist = []
        for i in range(self.history_len):
            h_a, dist_a = self.actor.batch_forward(torch.cat([states_a, no_info], dim=1), h_a)
            h_a = torch.permute(h_a, (1, 0, 2))
            dist_a = dist_a.reshape(self.batch_size, -1)

            action_b, dist_b = agent.select_action(env)
            action_a = torch.multinomial(dist_a, 1).reshape(self.batch_size)
            
            obs, r, _, _ = self.action_models.step([action_a, action_b])
            obs_a, obs_b = obs
            obs_a = obs_a.reshape(self.batch_size, -1)
            
            qa_hist.append(torch.cat([obs_a, dist_b], dim=1))

        qa_hist = torch.permute(torch.stack(qa_hist), (1, 0, 2))
        agent_r = self.qa_module(qa_hist)
        return agent_r

    def get_agent_representation(self, state_a, state_b, agent, h_a, h_b):
        no_info = torch.zeros(self.representation_size).to(self.device)
        qa_hist = []
        
        for i in range(self.history_len):
            h_a, dist_a = self.actor(torch.cat([state_a, no_info]), h_a)
            h_b, dist_b = agent.actor(torch.cat([state_b, no_info]), h_b)
            action_a = torch.multinomial(dist_a, 1)
            action_b = torch.multinomial(dist_b, 1)
            
            obs, r, _, _ = self.action_model.step([action_a, action_b])
            obs_a, obs_b = obs

            state_a = obs_a.flatten()
            state_b = obs_b.flatten()
            
            qa_hist.append(torch.cat([obs_a.flatten(), dist_b]))

        qa_hist = torch.stack(qa_hist).reshape(1, self.history_len, -1)
        agent_r = self.qa_module(qa_hist)
        return agent_r
    
    def get_agent_representations(self, states_a, states_b, agent, h_a, h_b):
        no_info =  torch.zeros(self.batch_size, self.representation_size).to(self.device)
        qa_hist = []
        # For dice operator (stochastic differentiation)
        log_probs_b = []
        for i in range(self.history_len):
            h_a, dist_a = self.actor.batch_forward(torch.cat([states_a, no_info], dim=1), h_a)
            h_b, dist_b = agent.actor.batch_forward(torch.cat([states_b, no_info], dim=1), h_b)

            h_a = torch.permute(h_a, (1, 0, 2))
            h_b = torch.permute(h_b, (1, 0, 2))
            dist_a = dist_a.reshape(self.batch_size, -1)
            dist_b = dist_b.reshape(self.batch_size, -1)

            actions_a = torch.multinomial(dist_a, 1).reshape(self.batch_size)
            actions_b = torch.multinomial(dist_b, 1).reshape(self.batch_size)

            b_t_probs = dist_b.gather(1, actions_b.reshape(-1, 1)).reshape(self.batch_size)
            log_probs_b.append(torch.log(b_t_probs))
            
            obs, r, _, _ = self.action_models.step([actions_a, actions_b])
            obs_a, obs_b = obs

            states_a = obs_a.reshape(self.batch_size, -1)
            states_b = obs_b.reshape(self.batch_size, -1)
            
            qa_hist.append(torch.cat([obs_a.reshape(self.batch_size, -1), dist_b], dim=1))

        qa_hist = torch.permute(torch.stack(qa_hist), (1, 0, 2))
        causal_logps = torch.cumsum(torch.permute(torch.stack(log_probs_b), (1, 0)), dim=1)
        causal_logps = causal_logps.unsqueeze(2).repeat(1, 1, qa_hist.shape[2]).reshape(self.batch_size, self.history_len, -1)
        agent_r = self.qa_module(magic_box(causal_logps) * qa_hist)
        return agent_r
    
    def select_action(self, state_a, state_b, agent, h_a, h_b):
        self.steps_done += 1
        agent_r = self.get_agent_representation(state_a, state_b, agent, h_a, h_b)
        h_a_cond, dist_a = self.actor(torch.cat([state_a, agent_r.flatten()]), h_a)
        action = torch.multinomial(dist_a, 1)
        return h_a_cond, action, agent_r
    
    def select_actions(self, states_a, states_b, agent, h_a, h_b, conditioned=True):
        self.steps_done += 1
        if conditioned:
            agent_r = self.get_agent_representations(states_a, states_b, agent, h_a, h_b)
        else:
            agent_r = torch.zeros(self.batch_size, self.representation_size).to(self.device)
        h_a_cond, dists_a = self.actor.batch_forward(torch.cat([states_a, agent_r.reshape(self.batch_size, -1)], dim=1), h_a)
        actions = torch.multinomial(dists_a.reshape(self.batch_size, -1), 1).reshape(self.batch_size)
        h_a_cond = torch.permute(h_a_cond, (1, 0, 2))
        return h_a_cond, actions, agent_r
    
    def compute_kl_divergence(self, state_a, agent_r, h_a):
        no_info = torch.zeros(self.representation_size).to(self.device)
        _, pi_info = self.actor(torch.cat([state_a, agent_r.flatten()]), h_a)
        _, pi_no_info = self.actor(torch.cat([state_a, no_info]), h_a)
        kl_1 = torch.sum(pi_info * torch.log(torch.div(pi_info, (pi_no_info+pi_info)/2)))
        kl_2 = torch.sum(pi_no_info * torch.log(torch.div(pi_no_info, (pi_no_info+pi_info)/2)))
        return (kl_1+kl_2)/2
    
    def compute_kl_divergences(self, agent, states_a, states_b, h_a, h_b):
        agent_r = self.get_agent_representations(states_a, states_b, agent, h_a, h_b)
        no_info = torch.zeros(self.batch_size, self.representation_size).to(self.device)
        _, pi_info = self.actor.batch_forward(torch.cat([states_a, agent_r], dim=1), h_a)
        _, pi_no_info = self.actor.batch_forward(torch.cat([states_a, no_info], dim=1), h_a)
        pi_info = pi_info.reshape(self.batch_size, -1)
        pi_no_info = pi_no_info.reshape(self.batch_size, -1)
        kl_1 = torch.mean(torch.sum(pi_info.detach() * torch.log(torch.div(pi_info.detach(), (pi_no_info+pi_info)/2)), dim=1))
        # kl_2 = torch.mean(torch.sum(pi_no_info * torch.log(torch.div(pi_no_info, (pi_no_info+pi_info)/2)), dim=1))
        return kl_1

    def compute_entropy_normalized(self, state_a, agent_r, h_a):
        _, pi = self.actor(torch.cat([state_a, agent_r.flatten()]), h_a)
        ent = -1 * torch.sum(pi * torch.log(pi))
        max_ent = torch.log(torch.ones(1)*self.n_actions).to(self.device)
        return ent/max_ent
    
    def compute_entropies_normalized(self, state_a, agent_r, h_a):
        _, pi = self.actor(torch.cat([state_a, agent_r.flatten()]), h_a)
        ent = -1 * torch.sum(pi * torch.log(pi))
        max_ent = torch.log(torch.ones(1)*self.n_actions).to(self.device)
        return ent/max_ent

    def compute_pg_loss(self, agent, agent_t=1, greedy=False):
        self.cum_steps = self.cum_steps + 1
        steps = self.cum_steps

        # Parallel Monte-carlo rollouts
        t_rewards = []
        a_rewards = []
        o_rewards = []
        log_probs_a = []
        log_probs_b = []
        obs_a, obs_b, h_a, h_b = self.transition

        rep_dropout = torch.tensor(self.rep_dropout).repeat(self.batch_size).to(self.device)

        states_a = obs_a.reshape((self.batch_size, -1))
        states_b = obs_b.reshape((self.batch_size, -1))
        reps_a = torch.mean(self.get_agent_representations(states_a, states_b, agent, h_a, h_b), dim=0)
        reps_b = torch.mean( self.get_agent_representations(states_b, states_a, self, h_b, h_a), dim=0)
        reps_a = reps_a.repeat(self.batch_size, 1)
        reps_b = reps_b.repeat(self.batch_size, 1)

        for i in range(self.rollout_len):
            self.action_models.clone_env_batch(self.model)

            states_a = obs_a.reshape((self.batch_size, -1))
            states_b = obs_b.reshape((self.batch_size, -1))

            reps_a_mask = torch.bernoulli(rep_dropout).unsqueeze(1).repeat(1, self.representation_size)
            reps_b_mask = torch.bernoulli(rep_dropout).unsqueeze(1).repeat(1, self.representation_size)

            h_a, dists_a = self.actor.batch_forward(torch.cat([states_a, reps_a_mask*reps_a], dim=1), h_a)
            h_b, dists_b = agent.actor.batch_forward(torch.cat([states_b, reps_b_mask*reps_b], dim=1), h_b)

            h_a = torch.permute(h_a, (1, 0, 2))
            h_b = torch.permute(h_b, (1, 0, 2))
            dists_a = dists_a.reshape((self.batch_size, -1))
            dists_b = dists_b.reshape((self.batch_size, -1))

            actions_a = torch.multinomial(dists_a, 1).reshape(self.batch_size)
            actions_b = torch.multinomial(dists_b, 1).reshape(self.batch_size)
            
            a_t_probs = dists_a.gather(1, actions_a.reshape(-1, 1)).reshape(self.batch_size)
            b_t_probs = dists_b.gather(1, actions_b.reshape(-1, 1)).reshape(self.batch_size)

            if greedy:
                b_t_probs = b_t_probs.detach()

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

            r1_reg = r1 - self.entropy_weight * torch.log(a_t_probs).detach()

            obs_a = obs_a.reshape((self.batch_size, -1))
            obs_b = obs_b.reshape((self.batch_size, -1))
            
            t_rewards.append(r1_reg)
            
            steps = steps + 1
        gammas = torch.tensor(self.gamma).repeat(self.batch_size, self.rollout_len - 1).to(self.device)
        gammas = torch.exp(torch.cumsum(torch.log(gammas), dim=1))
        gammas = torch.cat([torch.ones(self.batch_size, 1).to(self.device), gammas], dim=1)
        rewards_t = torch.permute(torch.stack(t_rewards), (1, 0))
        returns = torch.sum(rewards_t*gammas, dim=1)
        import pdb; pdb.set_trace()
        log_probs_c = torch.permute(torch.stack(log_probs_a) + torch.stack(log_probs_b), (1, 0))
        sum_log_probs = torch.sum(log_probs_c, dim=1)
        pg_loss = torch.mean(returns * sum_log_probs)

        return pg_loss
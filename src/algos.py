import hydra
import random
import torch
import wandb
import numpy as np

from itertools import count
from multiprocessing import Pool
from typing import Any, Dict, Optional

from .optimizers import ExtraAdam
from .utils import WandbLogger

def optimize_pg_loss(opt_type, opt_1, opt_2, loss_1, loss_2, t, grad_max_norm, agent_1, agent_2):
    if opt_type == "sgd" or opt_type == "adam":
        opt_1.zero_grad()
        opt_2.zero_grad()
        loss_1.backward(retain_graph=True)
        loss_2.backward()
        opt_1.step()
        opt_2.step()
    elif opt_type == "eg":
        loss_1 = -1 * loss_1
        loss_2 = -1 * loss_2 
        opt_1.zero_grad()
        opt_2.zero_grad()
        loss_1.backward(retain_graph=True)
        loss_2.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm(list(agent_1.actor.parameters()) + 
        #                               list(agent_1.history_aggregator.parameters()), 
        #                               grad_max_norm)
        # torch.nn.utils.clip_grad_norm(list(agent_2.actor.parameters()) + 
        #                               list(agent_2.history_aggregator.parameters()), 
        #                               grad_max_norm)
        if t % 2 == 0:
            opt_1.extrapolation()
            opt_2.extrapolation()
        else:
            opt_1.step()
            opt_2.step()

def get_metrics(env):
    adv_1, adv_2, em_1, em_2 = None, None, None, None
    if env.red_can_blue.detach():
        adv_1 = 1 if env.red_takes_blue.detach() else 0
    if env.blue_can_red.detach():
        adv_2 = 1 if env.blue_takes_red.detach() else 0
    if env.red_can_red.detach():
        em_1 = 0 if  env.red_takes_red.detach() else 1
    if env.blue_can_blue.detach():
        em_2 = 0 if  env.blue_takes_blue.detach() else 1
    return adv_1, adv_2, em_1, em_2

def evaluate_agents(agent_1, agent_2, evaluation_steps, eval_env, env_type, device):
    pass
    
def evaluate_agent(agent, evaluation_steps, env, env_type, device, pi):
    pass

def run_vip(env,
            eval_env,
            obs, 
            agent_1, 
            agent_2,  
            reward_window, 
            device,
            num_episodes,
            n_actions,
            evaluate_every=10,
            evaluation_steps=10,
            grad_max_norm=1,
            kl_weight=1,
            sp_weight=1):
    torch.backends.cudnn.benchmark = True
    logger = WandbLogger(device, reward_window)
    steps_reset = agent_1.rollout_len
    exploit_weight = 1
    for i_episode in range(num_episodes):
        obs, _ = env.reset()
        obs_1 = obs
        obs_2 =  torch.clone(obs[:, torch.tensor([1, 0, 3, 2])])
        
        for t in count():
            if t % steps_reset == 0:
                h_1, h_2 = None, None
                last_1 = -1*torch.ones(n_actions).to(device)
                last_2 = -1*torch.ones(n_actions).to(device)
            
            state_1 = torch.cat([obs_1.flatten(), last_1, last_2])
            state_2 = torch.cat([obs_2.flatten(), last_2, last_1])
            agent_1.action_model.clone_env(env)
            agent_2.action_model.clone_env(env)
            
            h_1_cond, action_1, rep_1 = agent_1.select_action(state_1, state_2, agent_2, h_1, h_2)
            h_2_cond, action_2, rep_2 = agent_2.select_action(state_2, state_1, agent_1, h_2, h_1)

            last_1 = torch.zeros(n_actions).to(device)
            last_2 = torch.zeros(n_actions).to(device)
            last_1 = last_1.scatter(0, action_1, 1)
            last_2 = last_2.scatter(0, action_2, 1)
            
            obs, r, _, _ = env.step([action_1, action_2])
            obs_1, obs_2 = obs
            r1, r2 = r

            adv_1, adv_2, em_1, em_2 = get_metrics(env)

            agent_1.transition = [obs_1, obs_2, h_1_cond, h_2_cond, last_1, last_2]
            agent_2.transition = [obs_2, obs_1, h_2_cond, h_1_cond, last_2, last_1]

            agent_1.model.clone_env(env)
            agent_2.model.clone_env(env)

            pg_loss_1 = agent_1.compute_pg_loss(agent_2, agent_t=1)
            pg_loss_2 = agent_2.compute_pg_loss(agent_1, agent_t=2)

            #sp_loss_1 = agent_1.compute_pg_loss(agent_1, agent_t=1)
            #sp_loss_2 = agent_2.compute_pg_loss(agent_2, agent_t=2)

            kl_1 = agent_1.compute_kl_divergence(state_1, rep_1, h_1_cond)
            kl_2 = agent_2.compute_kl_divergence(state_2, rep_2, h_2_cond)

            ent_1 = agent_1.compute_entropy_normalized(state_1, rep_1, h_1_cond)
            ent_2 = agent_2.compute_entropy_normalized(state_2, rep_2, h_2_cond)

            loss_1 = pg_loss_1 
            loss_2 = pg_loss_2

            optimize_pg_loss(agent_1.opt_type, 
                             agent_1.optimizer, 
                             agent_2.optimizer,
                             loss_1,
                             loss_2,
                             t,
                             grad_max_norm,
                             agent_1,
                             agent_2)

            d_1, c_1, d_2, c_2 = None, None, None, None

            logger.log_wandb_info(agent_1,
                                  agent_2,
                                  action_1, 
                                  action_2, 
                                  r1, 
                                  r2, 
                                  pg_loss_1, 
                                  pg_loss_2,
                                  device,
                                  d_score_1=d_1,
                                  c_score_1=c_1,
                                  d_score_2=d_2,
                                  c_score_2=c_2,
                                  obs=obs_1,
                                  kl_1=kl_1,
                                  kl_2=kl_2,
                                  adv_1=adv_1,
                                  adv_2=adv_2,
                                  em_1=em_1,
                                  em_2=em_2,
                                  ent_1=ent_1,
                                  ent_2=ent_2)
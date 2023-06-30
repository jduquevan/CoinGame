import hydra
import numpy as np
import random
import torch
import wandb

from itertools import count
from multiprocessing import Pool
from typing import Any, Dict, Optional

from .optimizers import ExtraAdam
from .utils import WandbLogger, get_metrics

def optimize_pg_loss(opt_type, opt_1, opt_2, loss_1, loss_2, t):
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
        if t % 2 == 0:
            opt_1.extrapolation()
            opt_2.extrapolation()
        else:
            opt_1.step()
            opt_2.step()

def evaluate_agents(agent_1, agent_2, a_c, a_d, evaluation_steps, eval_env, batch_size, conditioned=True):
    agent_1.eval()
    agent_2.eval()
    c_1 = evaluate_agent_fixed(agent_1, a_c, evaluation_steps, eval_env, batch_size, conditioned)
    c_2 = evaluate_agent_fixed(agent_2, a_c, evaluation_steps, eval_env, batch_size, conditioned)
    d_1 = evaluate_agent_fixed(agent_1, a_d, evaluation_steps, eval_env, batch_size, conditioned)
    d_2 = evaluate_agent_fixed(agent_2, a_d, evaluation_steps, eval_env, batch_size, conditioned)
    a_1 = evaluate_agent(agent_1, evaluation_steps, eval_env, batch_size, conditioned)
    a_2 = evaluate_agent(agent_2, evaluation_steps, eval_env, batch_size, conditioned)
    agent_1.train()
    agent_2.train()
    return c_1, c_2, d_1, d_2, a_1, a_2

def evaluate_agent_fixed(agent, fixed_agent, evaluation_steps, env, batch_size, conditioned):
    h_a = None
    rewards = []
    no_info = torch.zeros(batch_size, agent.representation_size).to(agent.device)
    obs, _ = env.reset()
    obs_a = obs.reshape(batch_size, -1)
    agent.action_models.clone_env_batch(env)
    for i in range(evaluation_steps):
        if conditioned:
            agent_r = agent.get_fixed_representations(obs_a, fixed_agent, h_a, env)
        else:
            agent_r = no_info
        h_a, dist_a = agent.actor.batch_forward(torch.cat([obs_a, agent_r], dim=1), h_a)
        h_a = torch.permute(h_a, (1, 0, 2))
        dist_a = dist_a.reshape(batch_size, -1)

        action_a = torch.multinomial(dist_a, 1).reshape(batch_size)
        action_b, dist_b = fixed_agent.select_action(env)

        obs, r, _, _ = env.step([action_a, action_b])
        obs_a, obs_b = obs
        obs_a = obs_a.reshape(batch_size, -1)
        r1, r2 = r
        rewards.append(r1)
    
    reward = torch.mean(torch.stack(rewards))
    return reward

def evaluate_agent(agent, evaluation_steps, env, batch_size, conditioned=True):
    h_a = None
    h_b = None
    rewards = []
    no_info = torch.zeros(batch_size, agent.representation_size).to(agent.device)
    obs, _ = env.reset()
    obs_a = obs.reshape(batch_size, -1)
    obs_b = torch.clone(obs[:, torch.tensor([1, 0, 3, 2])]).reshape(batch_size, -1)
    agent.action_models.clone_env_batch(env)
    for i in range(evaluation_steps):
        if conditioned:
            reps_a = agent.get_agent_representations(obs_a, obs_b, agent, h_a, h_b)
            reps_b = agent.get_agent_representations(obs_b, obs_a, agent, h_b, h_a)
        else:
            reps_a = no_info
            reps_b = no_info

        h_a, dist_a = agent.actor.batch_forward(torch.cat([obs_a, reps_a], dim=1), h_a)
        h_b, dist_b = agent.actor.batch_forward(torch.cat([obs_b, reps_b], dim=1), h_b)

        h_a = torch.permute(h_a, (1, 0, 2))
        h_b = torch.permute(h_b, (1, 0, 2))
        dist_a = dist_a.reshape(batch_size, -1)
        dist_b = dist_b.reshape(batch_size, -1)

        action_a = torch.multinomial(dist_a, 1).reshape(batch_size)
        action_b = torch.multinomial(dist_b, 1).reshape(batch_size)

        obs, r, _, _ = env.step([action_a, action_b])
        obs_a, obs_b = obs
        obs_a = obs_a.reshape(batch_size, -1)
        obs_b = obs_b.reshape(batch_size, -1)
        r1, r2 = r
        rewards.append(r1)
    
    reward = torch.mean(torch.stack(rewards))
    return reward

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
            sp_weight=1,
            always_cooperate=None,
            always_defect=None,
            greedy_p=0.5,
            batch_size=1):

    torch.backends.cudnn.benchmark = True
    logger = WandbLogger(device, reward_window)
    steps_reset = agent_1.rollout_len
    exploit_weight = 1
    c_1, c_2, d_1, d_2 = None, None, None, None

    for i_episode in range(num_episodes):
        obs, _ = env.reset()
        obs_1 = obs
        obs_2 =  torch.clone(obs[:, torch.tensor([1, 0, 3, 2])])
        
        for t in count():
            if t % steps_reset == 0:
                h_1, h_2 = None, None

            state_1 = obs_1.reshape(batch_size, -1)
            state_2 = obs_2.reshape(batch_size, -1)
            agent_1.action_models.clone_env_batch(env)
            agent_2.action_models.clone_env_batch(env)
           
            h_1_cond, action_1, rep_1 = agent_1.select_actions(state_1, state_2, agent_2, h_1, h_2, False)
            h_2_cond, action_2, rep_2 = agent_2.select_actions(state_2, state_1, agent_1, h_2, h_1, False)

            kl_1 = agent_1.compute_kl_divergences(state_1, rep_1, h_1_cond)
            kl_2 = agent_2.compute_kl_divergences(state_2, rep_2, h_2_cond)
            
            obs, r, _, _ = env.step([action_1, action_2])
            obs_1, obs_2 = obs
            r1, r2 = r
            
            adv_1, adv_2, em_1, em_2 = get_metrics(env)

            agent_1.transition = [obs_1, obs_2, h_1_cond, h_2_cond]
            agent_2.transition = [obs_2, obs_1, h_2_cond, h_1_cond]

            agent_1.model.clone_env_batch(env)
            agent_2.model.clone_env_batch(env)

            greedy_1 = np.random.binomial(1, greedy_p)
            greedy_2 = np.random.binomial(1, greedy_p)
            
            pg_loss_1, t11, t12 = agent_1.compute_pg_loss(agent_2, agent_t=1, greedy=greedy_1)
            pg_loss_2, t21, t22 = agent_2.compute_pg_loss(agent_1, agent_t=2, greedy=greedy_2)

            ent_1, ent_2 = None, None

            loss_1 = pg_loss_1 - kl_weight * kl_1
            loss_2 = pg_loss_2 - kl_weight * kl_2
    
            optimize_pg_loss(agent_1.opt_type, 
                             agent_1.optimizer, 
                             agent_2.optimizer,
                             loss_1,
                             loss_2,
                             t)

            if t % evaluate_every == 0:
                c_1, c_2, d_1, d_2, a_1, a_2 = evaluate_agents(agent_1, 
                                                               agent_2, 
                                                               always_cooperate,
                                                               always_defect,
                                                               evaluation_steps, 
                                                               eval_env,
                                                               batch_size)
                
                uc_1, uc_2, ud_1, ud_2, ua_1, ua_2 = evaluate_agents(agent_1, 
                                                                     agent_2, 
                                                                     always_cooperate,
                                                                     always_defect,
                                                                     evaluation_steps, 
                                                                     eval_env,
                                                                     batch_size,
                                                                     False)

            logger.log_wandb_info(agent_1,
                                  agent_2,
                                  action_1, 
                                  action_2, 
                                  torch.mean(r1).detach(), 
                                  torch.mean(r2).detach(), 
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
                                  ent_2=ent_2,
                                  a_1=a_1,
                                  a_2=a_2,
                                  uc_1=uc_1,
                                  uc_2=uc_2,
                                  ud_1=ud_1,
                                  ud_2=ud_2,
                                  ua_1=ua_1,
                                  ua_2=ua_2)
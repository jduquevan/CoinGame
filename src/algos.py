import hydra
import numpy as np
import random
import torch
import wandb

from itertools import count
from multiprocessing import Pool
from typing import Any, Dict, Optional
from torch.optim.lr_scheduler import MultiStepLR

from .optimizers import ExtraAdam
from .utils import (WandbLogger, 
                    compute_entropy, 
                    get_metrics, 
                    load_state_dict, 
                    magic_box,
                    compute_ipd_probs, 
                    add_gaussian_noise)

def optimize_loss(opt_type, optimizer, loss, t, scheduler, maximize=True):
    if opt_type == "sgd" or opt_type == "adam":
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss.detach_()
    elif opt_type == "eg":
        if maximize:
            loss = -1 * loss
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if t % 2 == 0:
            optimizer.extrapolation()
        else:
            optimizer.step()
    elif opt_type == "om":
        if maximize:
            loss = -1 * loss
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()

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
    last_actions_a = torch.zeros(agent.batch_size, agent.n_actions).to(agent.device)
    last_actions_b = torch.zeros(agent.batch_size, agent.n_actions).to(agent.device)
    obs, _ = env.reset()
    ob_a = obs.reshape(batch_size, -1)
    state_a = torch.cat([ob_a, last_actions_a, last_actions_b], dim=1)

    for i in range(evaluation_steps):

        h_a, dist_a = agent.actor.batch_forward(state_a, h_a)
        h_a = torch.permute(h_a, (1, 0, 2))
        dist_a = dist_a.reshape(batch_size, -1)

        action_a = torch.multinomial(dist_a, 1).reshape(batch_size)
        last_actions_a = torch.nn.functional.one_hot(action_a, 4)
        action_b, dist_b = fixed_agent.select_action(env)

        obs, r, _, _ = env.step([action_a, action_b])
        ob_a, ob_b = obs
        ob_a = ob_a.reshape(batch_size, -1)
        r1, r2 = r
        rewards.append(r1)

        state_a = torch.cat([ob_a, last_actions_a, dist_b], dim=1)
    
    reward = torch.mean(torch.stack(rewards))
    return reward

def evaluate_agent(agent, evaluation_steps, env, batch_size, conditioned=True):
    # import pdb; pdb.set_trace()
    h_a = None
    h_b = None
    rewards = []
    last_actions_a = torch.zeros(agent.batch_size, agent.n_actions).to(agent.device)
    last_actions_b = torch.zeros(agent.batch_size, agent.n_actions).to(agent.device)
    obs, _ = env.reset()
    ob_a = obs.reshape(batch_size, -1)
    ob_b = torch.clone(obs[:, torch.tensor([1, 0, 3, 2])]).reshape(batch_size, -1)
    state_a = torch.cat([ob_a, last_actions_a, last_actions_b], dim=1)
    state_b = torch.cat([ob_b, last_actions_b, last_actions_a], dim=1)

    for i in range(evaluation_steps):

        h_a, dist_a = agent.actor.batch_forward(state_a, h_a)
        h_b, dist_b = agent.actor.batch_forward(state_b, h_b)

        h_a = torch.permute(h_a, (1, 0, 2))
        h_b = torch.permute(h_b, (1, 0, 2))
        dist_a = dist_a.reshape(batch_size, -1)
        dist_b = dist_b.reshape(batch_size, -1)

        action_a = torch.multinomial(dist_a, 1).reshape(batch_size)
        action_b = torch.multinomial(dist_b, 1).reshape(batch_size)

        last_actions_a = torch.nn.functional.one_hot(action_a, 4)
        last_actions_b = torch.nn.functional.one_hot(action_b, 4)

        obs, r, _, _ = env.step([action_a, action_b])
        ob_a, ob_b = obs
        ob_a = ob_a.reshape(batch_size, -1)
        ob_b = ob_b.reshape(batch_size, -1)
        r1, r2 = r
        rewards.append(r1)

        state_a = torch.cat([ob_a, last_actions_a, last_actions_b], dim=1)
        state_b = torch.cat([ob_b, last_actions_b, last_actions_a], dim=1)
    
    reward = torch.mean(torch.stack(rewards))
    return reward

def evaluate_ipd_agents(agent_a, agent_b, evaluation_steps, env):
    agent_a.eval()
    agent_b.eval()
    c_a = evaluate_against_fixed_ipd(env, evaluation_steps, agent_a, True)
    c_b = evaluate_against_fixed_ipd(env, evaluation_steps, agent_b, True)
    d_a = evaluate_against_fixed_ipd(env, evaluation_steps, agent_a, False)
    d_b = evaluate_against_fixed_ipd(env, evaluation_steps, agent_b, False)
    agent_a.train()
    agent_b.train()
    return c_a, c_b, d_a, d_b

def evaluate_against_fixed_ipd(env, rollout_len, agent, always_cooperate=True):
    rewards = []
    h_a = None
    states_a, _ = env.reset()
    for i in range(rollout_len):
        h_a, dists_a = agent.actor.batch_forward(states_a, h_a)

        h_a = torch.permute(h_a, (1, 0, 2))
        dists_a = dists_a.reshape((agent.batch_size, -1))

        actions_a = torch.multinomial(dists_a, 1).reshape(agent.batch_size)

        if always_cooperate:
            actions_b = torch.zeros(agent.batch_size, dtype=torch.long).to(agent.device)
        else:
            actions_b = torch.ones(agent.batch_size, dtype=torch.long).to(agent.device)

        actions_a = torch.nn.functional.one_hot(actions_a, 2)
        actions_b = torch.nn.functional.one_hot(actions_b, 2)

        obs, r, _, _= env.step([actions_a, actions_b])
        states_a, states_b = obs
        ra, rb = r

        rewards.append(ra)
    return torch.mean(torch.stack(rewards))


def get_ipd_trajectories(env, rollout_len, agent_a, agent_b, entropy_weight, is_cg=False):
    h_a, h_b, h_ac, h_at, h_bc, h_bt = None, None, None, None, None, None
    return_dict = {}

    log_probs_a = []
    log_probs_b = []
    rewards_a = []
    rewards_b = []
    obs_a = []
    obs_b = []
    distributions_a = []
    distributions_b = []
    action_probs_a = []
    action_probs_b = []
    values_a = []
    values_b = []
    targets_a = []
    targets_b = []
    hiddens_a = []
    hiddens_b = []
    causal_rewards_b = []

    if is_cg:
        last_actions_a = torch.zeros(agent_a.batch_size, agent_a.n_actions).to(agent_a.device)
        last_actions_b = torch.zeros(agent_b.batch_size, agent_b.n_actions).to(agent_b.device)
        obs, _ = env.reset()
        ob_a = obs.reshape(agent_a.batch_size, -1)
        ob_b = torch.clone(obs[:, torch.tensor([1, 0, 3, 2])]).reshape(agent_b.batch_size, -1)
        states_a = torch.cat([ob_a, last_actions_a, last_actions_b], dim=1)
        states_b = torch.cat([ob_b, last_actions_b, last_actions_a], dim=1)
    else:
        states_a, _ = env.reset()
        states_b =  torch.clone(states_a)

    for i in range(rollout_len):
        obs_a.append(states_a)
        obs_b.append(states_b)
        
        h_a, dists_a = agent_a.actor.batch_forward(states_a, h_a)
        h_b, dists_b = agent_b.actor.batch_forward(states_b, h_b)

        h_ac, val_a = agent_a.critic.batch_forward(states_a, h_ac)
        h_bc, val_b = agent_b.target.batch_forward(states_b, h_bc)

        h_at, tar_a = agent_a.target.batch_forward(states_a, h_at)
        h_bt, tar_b = agent_b.target.batch_forward(states_b, h_bt)

        values_a.append(val_a)
        values_b.append(val_b)
        targets_a.append(tar_a)
        targets_b.append(tar_b)

        h_a = torch.permute(h_a, (1, 0, 2))
        h_b = torch.permute(h_b, (1, 0, 2))
        h_ac = torch.permute(h_ac, (1, 0, 2))
        h_at = torch.permute(h_at, (1, 0, 2))
        h_bc = torch.permute(h_bc, (1, 0, 2))
        h_bt = torch.permute(h_bt, (1, 0, 2))

        hiddens_a.append(h_ac)
        hiddens_b.append(h_bc)

        dists_a = dists_a.reshape((agent_a.batch_size, -1))
        dists_b = dists_b.reshape((agent_b.batch_size, -1))

        distributions_a.append(dists_a)
        distributions_b.append(dists_b)

        actions_a = torch.multinomial(dists_a, 1).reshape(agent_a.batch_size)
        actions_b = torch.multinomial(dists_b, 1).reshape(agent_b.batch_size)

        a_t_probs = dists_a.gather(1, actions_a.reshape(-1, 1)).reshape(agent_a.batch_size)
        b_t_probs = dists_b.gather(1, actions_b.reshape(-1, 1)).reshape(agent_b.batch_size)

        log_probs_a.append(torch.log(a_t_probs))
        log_probs_b.append(torch.log(b_t_probs))

        action_probs_a.append(a_t_probs)
        action_probs_b.append(b_t_probs)
        
        if is_cg:
            last_actions_a = torch.nn.functional.one_hot(actions_a, 4)
            last_actions_b = torch.nn.functional.one_hot(actions_b, 4)
        else:
            actions_a = torch.nn.functional.one_hot(actions_a, 2)
            actions_b = torch.nn.functional.one_hot(actions_b, 2)

        obs, r, _, _= env.step([actions_a, actions_b])

        if is_cg:
            ob_a, ob_b = obs
            ob_a = ob_a.reshape(agent_a.batch_size, -1)
            ob_b = ob_b.reshape(agent_a.batch_size, -1)
            states_a = torch.cat([ob_a, last_actions_a, last_actions_b], dim=1)
            states_b = torch.cat([ob_b, last_actions_b, last_actions_a], dim=1)
        else:
            states_a, states_b = obs
        
        ra, rb = r
        
        if is_cg:
            ra_reg = ra - entropy_weight * (torch.log(a_t_probs)).detach()
        else:
            ra_reg = ra - entropy_weight * (torch.log(a_t_probs).reshape(agent_a.batch_size, 1, 1).detach())
            ra_reg = ra_reg.reshape(agent_a.batch_size)
            rb = rb.reshape(agent_b.batch_size)

        if i == rollout_len -1:
            rewards_a.append(tar_a.reshape(agent_a.batch_size).detach())
            rewards_b.append(val_b.reshape(agent_a.batch_size).detach())
        else:
            rewards_a.append(ra_reg)
            rewards_b.append(rb)

    log_probs_a_t = torch.permute(torch.stack(log_probs_a), (1, 0))
    rewards_b_t = torch.permute(torch.stack(rewards_b), (1, 0)).to(agent_a.device)
    for i in range(rollout_len):
        if i > 0:
            mask = torch.cat([torch.zeros(agent_a.batch_size, i), torch.ones(agent_a.batch_size, rollout_len-i)], 
                             dim=1).to(agent_a.device)
        else:
            mask = torch.ones(agent_a.batch_size, rollout_len).to(agent_a.device)
        causal_log_probs_a_t = torch.cumsum(mask * log_probs_a_t, 1)
        causal_rewards_b.append(magic_box(causal_log_probs_a_t)*rewards_b_t)
    
    return_dict["log_probs_a"] = log_probs_a
    return_dict["log_probs_b"] = log_probs_b
    return_dict["rewards_a"] = rewards_a
    return_dict["rewards_b"] = rewards_b
    return_dict["states_a"] = obs_a
    return_dict["states_b"] = obs_b
    return_dict["dists_a"] = distributions_a
    return_dict["dists_b"] = distributions_b
    return_dict["action_probs_a"] = action_probs_a
    return_dict["action_probs_b"] = action_probs_b
    return_dict["values_a"] = values_a
    return_dict["values_b"] = values_b
    return_dict["targets_a"] = targets_a
    return_dict["targets_b"] = targets_b
    return_dict["hiddens_a"] = hiddens_a
    return_dict["hiddens_b"] = hiddens_b
    return_dict["causal_rewards_b"] = causal_rewards_b

    return return_dict

def run_vip(env, 
            agent_a, 
            agent_b, 
            reward_window, 
            device, 
            target_update, 
            eval_every, 
            entropy_weight, 
            evaluation_steps, 
            is_cg=False, 
            always_cooperate=None,
            always_defect=None):
    
    logger = WandbLogger(device, reward_window)
    scheduler_a = MultiStepLR(agent_a.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)
    scheduler_b = MultiStepLR(agent_b.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)
    rollout_len = agent_a.rollout_len
    for t in count():
        if agent_a.noisy:
            agent_a.actor.sample_noise()
            agent_a.critic.sample_noise()
            agent_a.target.sample_noise()
        return_dict = get_ipd_trajectories(env, rollout_len, agent_a, agent_b, entropy_weight, is_cg)
        states_trajectory_a = return_dict["states_a"]

        entropy_a = compute_entropy(return_dict["dists_a"], agent_a.n_actions)
        val_loss_a = agent_a.compute_value_loss(return_dict["values_a"], return_dict["targets_a"], return_dict["rewards_a"])
        val_loss_b = agent_b.compute_value_loss(return_dict["values_b"], return_dict["targets_b"], return_dict["rewards_b"])
        optimize_loss(agent_a.critic_opt_type, agent_a.critic_optimizer, val_loss_a, t, scheduler_a, maximize=False)
        optimize_loss(agent_b.critic_opt_type, agent_b.critic_optimizer, val_loss_b, t, scheduler_b, maximize=False)
        
        pg_loss_a, pos_adv_ratio_a, pos_ret_ratio_a = agent_a.compute_reinforce_loss(return_dict["log_probs_a"],
                                                                                     return_dict["log_probs_b"],
                                                                                     return_dict["states_a"],
                                                                                     return_dict["rewards_a"],
                                                                                     return_dict["rewards_b"],
                                                                                     return_dict["hiddens_a"],
                                                                                     return_dict["values_b"],
                                                                                     return_dict["causal_rewards_b"],
                                                                                     is_cg,
                                                                                     is_cg)
        optimize_loss(agent_a.opt_type, agent_a.optimizer, pg_loss_a, t, scheduler_a)

        if agent_b.noisy:
            agent_b.actor.sample_noise()
            agent_b.critic.sample_noise()
            agent_b.target.sample_noise()
        return_dict = get_ipd_trajectories(env, rollout_len, agent_b, agent_a, entropy_weight, is_cg)
        states_trajectory_b = return_dict["states_a"]

        entropy_b = compute_entropy(return_dict["dists_a"], agent_b.n_actions)
        val_loss_b = agent_b.compute_value_loss(return_dict["values_a"], return_dict["targets_a"], return_dict["rewards_a"])
        val_loss_a = agent_b.compute_value_loss(return_dict["values_b"], return_dict["targets_b"], return_dict["rewards_b"])
        optimize_loss(agent_b.critic_opt_type, agent_b.critic_optimizer, val_loss_b, t, scheduler_b, maximize=False)
        optimize_loss(agent_a.critic_opt_type, agent_a.critic_optimizer, val_loss_a, t, scheduler_a, maximize=False)
        
        pg_loss_b, pos_adv_ratio_b, pos_ret_ratio_b = agent_b.compute_reinforce_loss(return_dict["log_probs_a"],
                                                                                     return_dict["log_probs_b"],
                                                                                     return_dict["states_a"],
                                                                                     return_dict["rewards_a"],
                                                                                     return_dict["rewards_b"],
                                                                                     return_dict["hiddens_a"],
                                                                                     return_dict["values_b"],
                                                                                     return_dict["causal_rewards_b"],
                                                                                     is_cg,
                                                                                     is_cg)
        optimize_loss(agent_b.opt_type, agent_b.optimizer, pg_loss_b, t, scheduler_b)

        if t % target_update == 0:
            agent_a.target.load_state_dict(agent_a.critic.state_dict())
            agent_b.target.load_state_dict(agent_b.critic.state_dict())

        if t % eval_every == 0:
            if is_cg:
                c_1, c_2, d_1, d_2, a_1, a_2 = evaluate_agents(agent_a, 
                                                               agent_b, 
                                                               always_cooperate,
                                                               always_defect,
                                                               evaluation_steps, 
                                                               env,
                                                               agent_a.batch_size)
                adv_1, adv_2, em_1, em_2 = get_metrics(env)
                ra = torch.mean(torch.stack(return_dict["rewards_b"])[:-1, :])
                rb = torch.mean(torch.stack(return_dict["rewards_a"])[:-1, :])

            else:
                return_dict = get_ipd_trajectories(env, rollout_len, agent_a, agent_b, entropy_weight=0)
                c_a, c_b, d_a, d_b = evaluate_ipd_agents(agent_a, agent_b, evaluation_steps, env)
                p_c_s_a, p_c_cc_a, p_c_cd_a, p_c_dc_a, p_c_dd_a = compute_ipd_probs(states_trajectory_a, agent_a.device)
                p_c_s_b, p_c_cc_b, p_c_cd_b, p_c_dc_b, p_c_dd_b = compute_ipd_probs(states_trajectory_b, agent_a.device)
                ra = torch.mean(torch.stack(return_dict["rewards_a"])[:-1, :])
                rb = torch.mean(torch.stack(return_dict["rewards_b"])[:-1, :]) 

        if is_cg:
            logger.log_wandb_info(agent_a,
                                  agent_b, 
                                  ra.detach(), 
                                  rb.detach(), 
                                  pg_loss_a, 
                                  pg_loss_b,
                                  device,
                                  d_score_1=d_1,
                                  c_score_1=c_1,
                                  d_score_2=d_2,
                                  c_score_2=c_2,
                                  adv_1=adv_1,
                                  adv_2=adv_2,
                                  em_1=em_1,
                                  em_2=em_2,
                                  ent_1=entropy_a.detach(),
                                  ent_2=entropy_b.detach(),
                                  a_1=a_1,
                                  a_2=a_2,
                                  val_loss_1=val_loss_a.detach(),
                                  val_loss_2=val_loss_b.detach())
        else:
            logger.log_wandb_ipd_info(r1=ra.detach(), 
                                      r2=rb.detach(), 
                                      ent_1=entropy_a.detach(), 
                                      ent_2=entropy_b.detach(),
                                      pg_loss_1=pg_loss_a.detach(),
                                      pg_loss_2=pg_loss_b.detach(),
                                      val_loss_1=val_loss_a.detach(),
                                      val_loss_2=val_loss_b.detach(),
                                      exp_ent_1=None,
                                      exp_ent_2=None,
                                      exp_r1=None,
                                      exp_r2=None,
                                      exp_loss_1=None,
                                      exp_loss_2=None,
                                      c_a=c_a.detach(),
                                      c_b=c_b.detach(),
                                      d_a=d_a.detach(),
                                      d_b=d_b.detach(),
                                      p_c_s_a=p_c_s_a.detach(),
                                      p_c_cc_a=p_c_cc_a.detach(),
                                      p_c_cd_a=p_c_cd_a.detach(),
                                      p_c_dc_a=p_c_dc_a.detach(),
                                      p_c_dd_a=p_c_dd_a.detach(),
                                      p_c_s_b=p_c_s_b.detach(),
                                      p_c_cc_b=p_c_cc_b.detach(),
                                      p_c_cd_b=p_c_cd_b.detach(),
                                      p_c_dc_b=p_c_dc_b.detach(),
                                      p_c_dd_b=p_c_dd_b.detach(),
                                      pos_adv_ratio_a=pos_adv_ratio_a.detach(),
                                      pos_ret_ratio_a=pos_ret_ratio_a.detach(),
                                      pos_adv_ratio_b=pos_adv_ratio_b.detach(),
                                      pos_ret_ratio_b=pos_ret_ratio_b.detach())
        
def run_vip_v2(env, 
               agent_a, 
               agent_b, 
               reward_window, 
               device, 
               target_update, 
               eval_every, 
               entropy_weight, 
               evaluation_steps, 
               is_cg=False, 
               always_cooperate=None,
               always_defect=None):
    
    logger = WandbLogger(device, reward_window)
    scheduler_a = MultiStepLR(agent_a.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)
    scheduler_b = MultiStepLR(agent_b.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)
    rollout_len = agent_a.rollout_len
    for t in count():
        return_dict = get_ipd_trajectories(env, rollout_len, agent_a, agent_b, entropy_weight, is_cg)
        states_trajectory_a = return_dict["states_a"]

        entropy_a = compute_entropy(return_dict["dists_a"], agent_a.n_actions)
        val_loss_a = agent_a.compute_value_loss(return_dict["values_a"], return_dict["targets_a"], return_dict["rewards_a"])
        val_loss_b = agent_b.compute_value_loss(return_dict["values_b"], return_dict["targets_b"], return_dict["rewards_b"])
        optimize_loss(agent_a.critic_opt_type, agent_a.critic_optimizer, val_loss_a, t, scheduler_a, maximize=False)
        optimize_loss(agent_b.critic_opt_type, agent_b.critic_optimizer, val_loss_b, t, scheduler_b, maximize=False)
        
        pg_loss_a, pos_adv_ratio_a, pos_ret_ratio_a = agent_a.compute_reinforce_loss(return_dict["log_probs_a"],
                                                                                     return_dict["log_probs_b"],
                                                                                     return_dict["states_a"],
                                                                                     return_dict["rewards_a"],
                                                                                     return_dict["rewards_b"],
                                                                                     return_dict["hiddens_a"],
                                                                                     return_dict["values_b"],
                                                                                     return_dict["causal_rewards_b"],
                                                                                     is_cg)
        optimize_loss(agent_a.opt_type, agent_a.optimizer, pg_loss_a, t, scheduler_a)
        
        if t % target_update == 0:
            agent_b.actor.load_state_dict(agent_a.actor.state_dict())
            agent_b.critic.load_state_dict(agent_a.critic.state_dict())

            agent_a.target.load_state_dict(agent_a.critic.state_dict())
            agent_b.target.load_state_dict(agent_a.critic.state_dict())

        if t % eval_every == 0:
            c_1, c_2, d_1, d_2, a_1, a_2 = evaluate_agents(agent_a, 
                                                           agent_b, 
                                                           always_cooperate,
                                                           always_defect,
                                                           evaluation_steps, 
                                                           env,
                                                           agent_a.batch_size)
            adv_1, adv_2, em_1, em_2 = get_metrics(env)
            ra = torch.mean(torch.stack(return_dict["rewards_b"])[:-1, :])
            rb = torch.mean(torch.stack(return_dict["rewards_a"])[:-1, :])


        logger.log_wandb_info(agent_a,
                              agent_b, 
                              ra.detach(), 
                              rb.detach(), 
                              pg_loss_a, 
                              None,
                              device,
                              d_score_1=d_1,
                              c_score_1=c_1,
                              d_score_2=d_2,
                              c_score_2=c_2,
                              adv_1=adv_1,
                              adv_2=adv_2,
                              em_1=em_1,
                              em_2=em_2,
                              ent_1=entropy_a.detach(),
                              ent_2=None,
                              a_1=a_1,
                              a_2=a_2)
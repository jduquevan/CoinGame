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
                    get_metrics, load_state_dict, 
                    magic_box,
                    compute_ipd_probs, 
                    add_gaussian_noise)

def optimize_losses(opt_type, opt_1, opt_2, loss_1, loss_2, t, scheduler):
    if opt_type == "sgd" or opt_type == "adam":
        opt_1.zero_grad()
        opt_2.zero_grad()
        loss_1.backward(retain_graph=True)
        loss_2.backward(retain_graph=True)
        opt_1.step()
        opt_2.step()
        loss_1.detach_()
        loss_2.detach_()
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
    elif opt_type == "om":
        loss_1 = -1 * loss_1
        loss_2 = -1 * loss_2
        opt_1.zero_grad()
        opt_2.zero_grad()
        loss_1.backward(retain_graph=True)
        loss_2.backward(retain_graph=True)
        opt_1.step()
        opt_2.step()
    scheduler.step()

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
    no_info = torch.zeros(batch_size, agent.representation_size).to(agent.device)
    obs, _ = env.reset()
    obs_a = obs.reshape(batch_size, -1)
    agent.action_models.clone_env_batch(env)
    for i in range(evaluation_steps):
        if conditioned:
            agent_r = agent.get_fixed_representations(obs_a, fixed_agent, h_a, env)
        else:
            agent_r = no_info

        state_a = obs_a.reshape((batch_size, -1))

        h_a, dist_a = agent.actor.batch_forward(torch.cat([state_a, agent_r], dim=1), h_a)
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

        state_a = obs_a.reshape((batch_size, -1))
        state_b = obs_b.reshape((batch_size, -1))

        h_a, dist_a = agent.actor.batch_forward(torch.cat([state_a, reps_a], dim=1), h_a)
        h_b, dist_b = agent.actor.batch_forward(torch.cat([state_b, reps_b], dim=1), h_b)

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

        obs, r, _= env.step([actions_a, actions_b])
        states_a, states_b = obs
        ra, rb = r

        rewards.append(ra)
    return torch.mean(torch.stack(rewards))


def get_ipd_trajectories(env, rollout_len, agent_a, agent_b, entropy_weight, evaluate=False):
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

        action_probs_a.append(a_t_probs)
        action_probs_b.append(b_t_probs)

        actions_a = torch.nn.functional.one_hot(actions_a, 2)
        actions_b = torch.nn.functional.one_hot(actions_b, 2)

        log_probs_a.append(torch.log(a_t_probs))
        log_probs_b.append(torch.log(b_t_probs))

        obs, r, _= env.step([actions_a, actions_b])
        states_a, states_b = obs
        ra, rb = r

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

def run_vip_ipd(env, agent_a, agent_b, reward_window, device, target_update, eval_every, entropy_weight, evaluation_steps):
    logger = WandbLogger(device, reward_window)
    scheduler_a = MultiStepLR(agent_a.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)
    scheduler_b = MultiStepLR(agent_b.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)
    rollout_len = agent_a.rollout_len
    for t in count():
        return_dict = get_ipd_trajectories(env, rollout_len, agent_a, agent_b, entropy_weight)
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
                                                                                     return_dict["causal_rewards_b"])
        optimize_loss(agent_a.opt_type, agent_a.optimizer, pg_loss_a, t, scheduler_a)

        return_dict = get_ipd_trajectories(env, rollout_len, agent_b, agent_a, entropy_weight)
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
                                                                                     return_dict["causal_rewards_b"])
        optimize_loss(agent_b.opt_type, agent_b.optimizer, pg_loss_b, t, scheduler_b)

        if t % target_update == 0:
            agent_a.target.load_state_dict(agent_a.critic.state_dict())
            agent_b.target.load_state_dict(agent_b.critic.state_dict())

        if t % eval_every == 0:
            return_dict = get_ipd_trajectories(env, rollout_len, agent_a, agent_b, entropy_weight=0, evaluate=True)
            ra = torch.mean(torch.stack(return_dict["rewards_a"])[:-1, :])
            rb = torch.mean(torch.stack(return_dict["rewards_b"])[:-1, :])
            c_a, c_b, d_a, d_b = evaluate_ipd_agents(agent_a, agent_b, evaluation_steps, env)
            p_c_s_a, p_c_cc_a, p_c_cd_a, p_c_dc_a, p_c_dd_a = compute_ipd_probs(states_trajectory_a, agent_a.device)
            p_c_s_b, p_c_cc_b, p_c_cd_b, p_c_dc_b, p_c_dd_b = compute_ipd_probs(states_trajectory_b, agent_a.device)

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
            batch_size=1,
            reset_every=1000):

    logger = WandbLogger(device, reward_window)
    steps_reset = agent_1.rollout_len
    exploit_weight = 1
    c_1, c_2, d_1, d_2 = None, None, None, None

    scheduler_1 = MultiStepLR(agent_1.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)
    scheduler_2 = MultiStepLR(agent_2.optimizer, milestones=[10000], gamma=0.5, last_epoch=-1, verbose=False)

    for i_episode in range(num_episodes):
        obs, _ = env.reset()
        obs_1 = obs
        obs_2 =  torch.clone(obs[:, torch.tensor([1, 0, 3, 2])])
        
        for t in count():
            if t % reset_every == 0:
                load_state_dict(agent_1.actor, "agent_1_init")
                load_state_dict(agent_2.actor, "agent_2_init")
            if t % steps_reset == 0:
                h_1, h_2 = None, None

            state_1 = obs_1.reshape((batch_size, -1))
            state_2 = obs_2.reshape((batch_size, -1))
            
            agent_1.action_models.clone_env_batch(env)
            agent_2.action_models.clone_env_batch(env)
           
            h_1, action_1, rep_1 = agent_1.select_actions(state_1, state_2, agent_2, h_1, h_2, False)
            h_2, action_2, rep_2 = agent_2.select_actions(state_2, state_1, agent_1, h_2, h_1, False)

            h_1 = torch.permute(h_1, (1, 0, 2))
            h_2 = torch.permute(h_2, (1, 0, 2))

            kl_1 = agent_1.compute_kl_divergences(agent_2, state_1, state_2, h_1, h_2)
            kl_2 = agent_2.compute_kl_divergences(agent_1, state_2, state_1, h_2, h_1)
            
            obs, r, _, _ = env.step([action_1, action_2])
            obs_1, obs_2 = obs
            r1, r2 = r
            
            adv_1, adv_2, em_1, em_2 = get_metrics(env)

            agent_1.transition = [obs_1, obs_2, h_1, h_2]
            agent_2.transition = [obs_2, obs_1, h_2, h_1]

            agent_1.model.clone_env_batch(env)
            agent_2.model.clone_env_batch(env)

            greedy_1 = np.random.binomial(1, greedy_p)
            greedy_2 = np.random.binomial(1, greedy_p)
            
            pg_loss_1, inf_loss_1 = agent_1.compute_pg_loss(agent_2, agent_t=1, greedy=greedy_1)
            loss_1 = pg_loss_1 - kl_weight * kl_1
            optimize_losses(agent_1.opt_type, agent_1.optimizer, agent_1.inf_optimizer, loss_1, inf_loss_1, t, scheduler_1)
            pg_loss_2, inf_loss_2 = agent_2.compute_pg_loss(agent_1, agent_t=2, greedy=greedy_2)
            loss_2 = pg_loss_2 - kl_weight * kl_2
            optimize_losses(agent_2.opt_type, agent_2.optimizer, agent_2.inf_optimizer, loss_2, inf_loss_2, t, scheduler_2)

            ent_1, ent_2 = None, None

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
import hydra
import random
import sys
import torch
import wandb
import numpy as np

from omegaconf import DictConfig, OmegaConf

from .agents import VIPAgent, VIPAgentIPD, VIPAgentIPDV2, AlwaysCooperateAgent, AlwaysDefectAgent
from .algos import run_vip, run_vip_ipd, run_vip_ipd_v2
from .coin_game import OGCoinGameGPU
from .ipd import IPD
from .utils import save_state_dict

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_path="../scripts", config_name="config", version_base=None)
def main(args: DictConfig):
    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    
    seed_all(config["seed"])

    wandb.init(config=config, dir=config["wandb_dir"], project="Co-games", reinit=True, anonymous="allow")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_type = config["env_type"]
    reward_window = config["reward_window"]
    num_episodes = config["num_episodes"]
    sp_weight = config["sp_weight"]
    kl_weight = config["kl_weight"]
    evaluate_every = config["evaluate_every"]
    evaluation_steps = config["evaluation_steps"]
    greedy_p = config["greedy_p"]

    if env_type == "cg":
        n_actions = 4
        batch_size = config["vip_agent"]["batch_size"]

        env = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
        eval_env = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
        model_1 = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
        action_models_1 = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
        model_2 = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
        action_models_2 = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
        obs, _ = env.reset()
        
        agent_1 = VIPAgent(config["base_agent"],
                           **config["vip_agent"],
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs[0].shape,
                           model=model_1,
                           action_models=action_models_1)
        agent_2 = VIPAgent(config["base_agent"],
                           **config["vip_agent"],
                           device=device, 
                           n_actions=n_actions,
                           obs_shape=obs[0].shape,
                           model=model_2,
                           action_models=action_models_2,
                           qa_module=agent_1.qa_module)
        
        save_state_dict(agent_1.actor, "agent_1_init")
        save_state_dict(agent_2.actor, "agent_2_init")
        always_cooperate = AlwaysCooperateAgent(False, device=device, n_actions=n_actions)
        always_defect = AlwaysDefectAgent(False, device=device, n_actions=n_actions)

        run_vip(env=env,
                eval_env=eval_env,
                obs=obs, 
                agent_1=agent_1, 
                agent_2=agent_2,  
                reward_window=reward_window, 
                device=device,
                num_episodes=num_episodes,
                n_actions=n_actions,
                evaluate_every=evaluate_every,
                kl_weight=kl_weight,
                sp_weight=sp_weight,
                always_cooperate=always_cooperate,
                always_defect=always_defect,
                greedy_p=greedy_p,
                batch_size=batch_size,
                reset_every=config["reset_every"])
        
    elif env_type == "ipd":
        n_actions = 2
        target_update = config["target_update"]
        exp_weight = config["exp_weight"]
        entropy_weight = config["entropy_weight"]
        batch_size = config["vip_agent_ipd"]["batch_size"]
        agent_ver = config["agent_ver"]

        env = IPD(device, batch_size)
        obs, _ = env.reset()

        if agent_ver == "v1":
            agent_1 = VIPAgentIPD(config["base_agent"],
                                **config["vip_agent_ipd"],
                                device=device,
                                n_actions=n_actions,
                                obs_shape=obs[0].shape)
            agent_2 = VIPAgentIPD(config["base_agent"],
                                **config["vip_agent_ipd"],
                                device=device,
                                n_actions=n_actions,
                                obs_shape=obs[0].shape)

            run_vip_ipd(env=env, 
                        agent_a=agent_1, 
                        agent_b=agent_2, 
                        reward_window=reward_window, 
                        device=device,
                        target_update=target_update,
                        exp_weight=exp_weight,
                        eval_every=evaluate_every,
                        entropy_weight=entropy_weight)
            
        elif agent_ver == "v2":
            agent_1 = VIPAgentIPDV2(config["base_agent"],
                                    **config["vip_agent_ipd_v2"],
                                    device=device,
                                    n_actions=n_actions,
                                    obs_shape=obs[0].shape)
            agent_2 = VIPAgentIPDV2(config["base_agent"],
                                    **config["vip_agent_ipd_v2"],
                                    device=device,
                                    n_actions=n_actions,
                                    obs_shape=obs[0].shape)

            run_vip_ipd_v2(env=env, 
                           agent_a=agent_1, 
                           agent_b=agent_2, 
                           reward_window=reward_window, 
                           device=device,
                           target_update=target_update,
                           eval_every=evaluate_every,
                           entropy_weight=entropy_weight)

if __name__ == "__main__":
    main()
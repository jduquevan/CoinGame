import hydra
import random
import sys
import torch
import wandb
import numpy as np

from omegaconf import DictConfig, OmegaConf

from .agents import VIPAgent, AlwaysCooperateAgent, AlwaysDefectAgent
from .algos import run_vip, run_vip_v2
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
    evaluate_every = config["evaluate_every"]
    evaluation_steps = config["evaluation_steps"]
    target_update = config["target_update"]
    entropy_weight = config["entropy_weight"]
    batch_size = config["vip_agent"]["batch_size"]

    if env_type == "cg":
        n_actions = 4
        version = config["version"]

        env = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
        obs, _ = env.reset()
        
        agent_1 = VIPAgent(config["base_agent"],
                           **config["vip_agent"],
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs[0].shape,
                           is_cg=True)
        agent_2 = VIPAgent(config["base_agent"],
                           **config["vip_agent"],
                           device=device, 
                           n_actions=n_actions,
                           obs_shape=obs[0].shape,
                           is_cg=True)
        
        always_cooperate = AlwaysCooperateAgent(False, device=device, n_actions=n_actions)
        always_defect = AlwaysDefectAgent(False, device=device, n_actions=n_actions)

        if version=="v1":
            run_vip(env=env, 
                    agent_a=agent_1, 
                    agent_b=agent_2, 
                    reward_window=reward_window, 
                    device=device,
                    target_update=target_update,
                    eval_every=evaluate_every,
                    entropy_weight=entropy_weight,
                    evaluation_steps=evaluation_steps,
                    is_cg=True,
                    always_cooperate=always_cooperate,
                    always_defect=always_defect)
        elif  version=="v2":
            run_vip_v2(env=env, 
                       agent_a=agent_1, 
                       agent_b=agent_1, 
                       reward_window=reward_window, 
                       device=device,
                       target_update=target_update,
                       eval_every=evaluate_every,
                       entropy_weight=entropy_weight,
                       evaluation_steps=evaluation_steps,
                       is_cg=True,
                       always_cooperate=always_cooperate,
                       always_defect=always_defect)
        
    elif env_type == "ipd":
        n_actions = 2

        env = IPD(device, batch_size)
        obs, _ = env.reset()
        
        agent_1 = VIPAgent(config["base_agent"],
                           **config["vip_agent"],
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs[0].shape,
                           is_cg=False)
        agent_2 = VIPAgent(config["base_agent"],
                           **config["vip_agent"],
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs[0].shape,
                           is_cg=False)

        run_vip(env=env, 
                agent_a=agent_1, 
                agent_b=agent_1, 
                reward_window=reward_window, 
                device=device,
                target_update=target_update,
                eval_every=evaluate_every,
                entropy_weight=entropy_weight,
                evaluation_steps=evaluation_steps,
                is_cg=False)

if __name__ == "__main__":
    main()
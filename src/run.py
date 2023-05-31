import hydra
import random
import sys
import torch
import wandb
import numpy as np

from omegaconf import DictConfig, OmegaConf

from .agents import VIPAgent
from .algos import run_vip
from .coin_game import OGCoinGameGPU

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_path="../scripts", config_name="config", version_base=None)
def main(args: DictConfig):
    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    
    seed_all(config["seed"])

    wandb.init(config=config, dir="/network/scratch/j/juan.duque/wandb/", project="Co-games", reinit=True, anonymous="allow")

    n_actions = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_window = config["reward_window"]
    num_episodes = config["num_episodes"]
    kl_weight = config["kl_weight"]
    batch_size = config["vip_agent"]["batch_size"]

    env = OGCoinGameGPU(**config["env"], batch_size=1, device=device)
    eval_env = OGCoinGameGPU(**config["env"], batch_size=1, device=device)
    model = OGCoinGameGPU(**config["env"], batch_size=batch_size, device=device)
    action_model = OGCoinGameGPU(**config["env"], batch_size=1, device=device)
    obs, _ = env.reset()
    
    agent_1 = VIPAgent(config["base_agent"],
                       **config["vip_agent"],
                       device=device,
                       n_actions=n_actions,
                       obs_shape=obs.shape,
                       model=model,
                       action_model=action_model)
    agent_2 = VIPAgent(config["base_agent"],
                       **config["vip_agent"],
                       device=device, 
                       n_actions=n_actions,
                       obs_shape=obs.shape,
                       model=model,
                       action_model=action_model)

    run_vip(env=env,
            eval_env=eval_env,
            obs=obs, 
            agent_1=agent_1, 
            agent_2=agent_2,  
            reward_window=reward_window, 
            device=device,
            num_episodes=config["num_episodes"],
            n_actions=n_actions,
            kl_weight=kl_weight)
        

if __name__ == "__main__":
    main()
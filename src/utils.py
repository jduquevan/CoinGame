import torch
import wandb

def magic_box(tau):
    return torch.exp(tau - tau.detach())

class WandbLogger():
    def __init__(self, device, reward_window):
        self.device = device
        self.reward_window = reward_window
        self.cum_steps = 0
        self.d_score_1 = 0
        self.d_score_2 = 0
        self.c_score_1 = 0
        self.c_score_2 = 0
        self.avg_reward_1 = []
        self.avg_reward_2 = []
        self.adversity_1 = []
        self.adversity_2 = []
        self.a_1_pickups = []
        self.a_2_pickups = []
        self.easy_misses_1 = []
        self.easy_misses_2 = []

    def log_wandb_info(self,
                       agent_1,
                       agent_2,
                       action_1, 
                       action_2,
                       r1, 
                       r2, 
                       pg_loss_1, 
                       pg_loss_2,
                       device,
                       d_score_1=None, 
                       c_score_1=None,
                       d_score_2=None,
                       c_score_2=None,
                       obs=None,
                       kl_1=None,
                       kl_2=None,
                       adv_1=None,
                       adv_2=None,
                       em_1=None,
                       em_2=None,
                       ent_1=None,
                       ent_2=None,
                       t_r1=None,
                       t_r2=None):
                       
        self.cum_steps = self.cum_steps + 1
        self.avg_reward_1.insert(0, r1)
        self.avg_reward_2.insert(0, r2)

        self.avg_reward_1 = self.avg_reward_1[0:self.reward_window]
        self.avg_reward_2 = self.avg_reward_2[0:self.reward_window]

        avg_1 = sum(self.avg_reward_1)/len(self.avg_reward_1)
        avg_2 = sum(self.avg_reward_2)/len(self.avg_reward_2)

        if d_score_1:
            d_score_1 = d_score_1.detach()
        if d_score_2:
            d_score_2 = d_score_2.detach()
        if c_score_1:
            c_score_1 = c_score_1.detach()
        if c_score_2:
            c_score_2 = c_score_2.detach()
        
        if adv_1 != None:
            self.adversity_1.append(adv_1)
        if adv_2 != None:
            self.adversity_2.append(adv_2)
        if em_1 != None:
            self.easy_misses_1.append(em_1)
        if em_2 != None:
            self.easy_misses_2.append(em_2)
        
        self.adversity_1 = self.adversity_1[-self.reward_window:]
        self.adversity_2 = self.adversity_2[-self.reward_window:]
        self.a_1_pickups = self.a_1_pickups[-self.reward_window:]
        self.a_2_pickups = self.a_2_pickups[-self.reward_window:]
        self.easy_misses_1 = self.easy_misses_1[-self.reward_window:]
        self.easy_misses_2 = self.easy_misses_2[-self.reward_window:]

        adv_1, adv_2, em_1, em_2 = None, None, None, None
        
        if len(self.adversity_1)==self.reward_window:
            adv_1 = sum(self.adversity_1)/len(self.adversity_1)
        if len(self.adversity_2)==self.reward_window:
            adv_2 = sum(self.adversity_2)/len(self.adversity_2)
        if len(self.easy_misses_1)==self.reward_window:
            em_1 = sum(self.easy_misses_1)/len(self.easy_misses_1)
        if len(self.easy_misses_2)==self.reward_window:
            em_2 = sum(self.easy_misses_2)/len(self.easy_misses_2)

        norms = []
        grad_norms = []
        for model in [agent_1.qa_module, agent_1.actor, agent_2.qa_module, agent_2.actor]:
            total_norm = 0
            total_grad_norm = 0
            for p in model.parameters():
                param_norm = p.detach().data.norm(2)
                grad_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm
                total_grad_norm += grad_norm
            norms.append(total_norm)
            grad_norms.append(total_grad_norm)

        wandb_info = {}
        wandb_info['qa_norm_1'] = norms[0]
        wandb_info['actor_norm_1'] = norms[1]
        wandb_info['qa_norm_2'] = norms[2]
        wandb_info['actor_norm_2'] = norms[3]
        wandb_info['qa_grad_norm_1'] = grad_norms[0]
        wandb_info['actor_grad_norm_1'] = grad_norms[1]
        wandb_info['qa_grad_norm_2'] = grad_norms[2]
        wandb_info['actor_grad_norm_2'] = grad_norms[3]
        wandb_info['cum_steps'] = self.cum_steps
        wandb_info['agent_1_avg_reward'] = avg_1
        wandb_info['agent_2_avg_reward'] = avg_2
        wandb_info['total_avg_reward'] = (avg_1 + avg_2)/2
        wandb_info['pg_loss_1'] = pg_loss_1
        wandb_info['pg_loss_2'] = pg_loss_2
        wandb_info['defect_score_1'] = d_score_1
        wandb_info['defect_score_2'] = d_score_2
        wandb_info['coop_score_1'] = c_score_1
        wandb_info['coop_score_2'] = c_score_2
        wandb_info['kl_1'] = kl_1
        wandb_info['kl_2'] = kl_2
        wandb_info['adversity_1'] = adv_1
        wandb_info['adversity_2'] = adv_2
        wandb_info['easy_misses_1'] = em_1
        wandb_info['easy_misses_2'] = em_2
        wandb_info['entropy_1'] = ent_1
        wandb_info['entropy_2'] = ent_2
        wandb_info['training_return_1'] = t_r1
        wandb_info['training_return_2'] = t_r2

        wandb.log(wandb_info)
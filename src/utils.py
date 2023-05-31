import wandb

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
                       action_1, 
                       action_2,
                       r1, 
                       r2, 
                       value_1, 
                       value_2,
                       d_score_1=None, 
                       c_score_1=None,
                       d_score_2=None,
                       c_score_2=None,
                       obs=None,
                       last_obs=None,
                       kl_1=None,
                       kl_2=None,
                       adv_1=None,
                       adv_2=None,
                       em_1=None,
                       em_2=None):
        self.cum_steps = self.cum_steps + 1
        self.avg_reward_1.insert(0, r1)
        self.avg_reward_2.insert(0, r2)

        self.avg_reward_1 = self.avg_reward_1[0:self.reward_window]
        self.avg_reward_2 = self.avg_reward_2[0:self.reward_window]

        avg_1 = sum(self.avg_reward_1)/len(self.avg_reward_1)
        avg_2 = sum(self.avg_reward_2)/len(self.avg_reward_2)

        if d_score_1:
            self.d_score_1 = d_score_1
        if d_score_2:
            self.d_score_2 = d_score_2
        if c_score_1:
            self.c_score_1 = c_score_1
        if c_score_2:
            self.c_score_2 = c_score_2
        
        if adv_1:
            self.adversity_1.append(adv_1)
            adv_1 = sum(self.adversity_1)/len(self.adversity_1)
        if adv_2:
            self.adversity_2.append(adv_2)
            adv_2 = sum(self.adversity_2)/len(self.adversity_2)
        if em_1:
            self.easy_misses_1.append(em_1)
            em_1 = sum(self.easy_misses_1)/len(self.easy_misses_1)
        if em_2:
            self.easy_misses_2.append(em_2)
            em_2 = sum(self.easy_misses_2)/len(self.easy_misses_2)
        
        self.adversity_1 = self.adversity_1[0:self.reward_window]
        self.adversity_2 = self.adversity_2[0:self.reward_window]
        self.a_1_pickups = self.a_1_pickups[0:self.reward_window]
        self.a_2_pickups = self.a_2_pickups[0:self.reward_window]
        self.easy_misses_1 = self.easy_misses_1[0:self.reward_window]
        self.easy_misses_2 = self.easy_misses_2[0:self.reward_window]

       
        wandb_info = {}
        wandb_info['cum_steps'] = self.cum_steps
        wandb_info['agent_1_avg_reward'] = avg_1
        wandb_info['agent_2_avg_reward'] = avg_2
        wandb_info['total_avg_reward'] = (avg_1 + avg_2)/2
        wandb_info['loss_1'] = value_1
        wandb_info['loss_2'] = value_2
        wandb_info['d_score_1'] = d_score_1
        wandb_info['d_score_2'] = d_score_2
        wandb_info['c_score_1'] = c_score_1
        wandb_info['c_score_2'] = c_score_2
        wandb_info['adversity_1'] = adv_1
        wandb_info['adversity_2'] = adv_2
        wandb_info['em_1'] = em_1
        wandb_info['em_2'] = em_2
        wandb_info['kl_1'] = kl_1
        wandb_info['kl_2'] = kl_2
        wandb_info['adversity_1'] = adv_1
        wandb_info['adversity_2'] = adv_2
        wandb_info['easy_misses_1'] = em_1
        wandb_info['easy_misses_2'] = em_2

        wandb.log(wandb_info)
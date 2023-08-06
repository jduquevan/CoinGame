import torch
import wandb

def magic_box(tau):
    return torch.exp(tau - tau.detach())

def add_gaussian_noise(model, weight, device):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()).to(device) * weight)

def compute_entropy(dists, n_actions):
    dists = torch.stack(dists).reshape((-1, n_actions))
    entropy = -torch.mean(torch.sum(dists * torch.log(dists), dim=1))
    return entropy

def save_state_dict(model, path):
    torch.save(model.state_dict(), './weights/'+ path +'.pt')

def load_state_dict(model, path):
    state_dict = torch.load('./weights/'+ path +'.pt')
    model.load_state_dict(state_dict)

def get_metrics(env):
    adv_1, adv_2, em_1, em_2 = None, None, None, None

    adv_1 = (torch.sum(torch.logical_and(env.red_can_blue, env.red_takes_blue))
             /torch.sum(env.red_can_blue)).detach()
    adv_2 = (torch.sum(torch.logical_and(env.blue_can_red, env.blue_takes_red))
             /torch.sum(env.blue_can_red)).detach()
    em_1 = torch.sum(env.red_can_red) - torch.sum(torch.logical_and(env.red_can_red, env.red_takes_red))
    em_1 = (em_1/torch.sum(env.red_can_red)).detach()
    em_2 = torch.sum(env.blue_can_blue) - torch.sum(torch.logical_and(env.blue_can_blue, env.blue_takes_blue))
    em_2 = (em_2/torch.sum(env.blue_can_blue)).detach()

    return adv_1, adv_2, em_1, em_2

def compute_ipd_probs(states, device):
    states = torch.permute(torch.stack(states), (1, 0, 2))

    start = torch.Tensor([1, 1, 1, 1]).float().repeat(states.shape[0], 1).to(device)
    cc = torch.Tensor([1, 0, 1, 0]).float().repeat(states.shape[0], 1).to(device)
    cd = torch.Tensor([1, 0, 0, 1]).float().repeat(states.shape[0], 1).to(device)
    dc = torch.Tensor([0, 1, 1, 0]).float().repeat(states.shape[0], 1).to(device)
    dd = torch.Tensor([0, 1, 0, 1]).float().repeat(states.shape[0], 1).to(device)
    coop = torch.Tensor([1, 0, 0, 0]).float().repeat(states.shape[0], 1).to(device)

    start_num = torch.zeros(1).to(device)
    cc_num = torch.zeros(1).to(device)
    cd_num = torch.zeros(1).to(device)
    dc_num = torch.zeros(1).to(device)
    dd_num = torch.zeros(1).to(device)

    c_start_num = torch.zeros(1).to(device)
    c_cc_num = torch.zeros(1).to(device)
    c_cd_num = torch.zeros(1).to(device)
    c_dc_num = torch.zeros(1).to(device)
    c_dd_num = torch.zeros(1).to(device)

    for i in range(states.shape[1] - 1):
        start_mask = torch.zeros_like(states).to(device)
        cc_mask = torch.zeros_like(states).to(device)
        cd_mask = torch.zeros_like(states).to(device)
        dc_mask = torch.zeros_like(states).to(device)
        dd_mask = torch.zeros_like(states).to(device)
        coop_mask = torch.zeros_like(states).to(device)
        
        start_mask[:, i, :] = start
        cc_mask[:, i, :] = cc
        cd_mask[:, i, :] = cd
        dc_mask[:, i, :] = dc
        dd_mask[:, i, :] = dd
        coop_mask[:, i+1, :] = coop

        if i > 0:
            cc_num += torch.sum((torch.sum(states*cc_mask, dim=2) > 1))
            cd_num += torch.sum((torch.sum(states*cd_mask, dim=2) > 1))
            dc_num += torch.sum((torch.sum(states*dc_mask, dim=2) > 1))
            dd_num += torch.sum((torch.sum(states*dd_mask, dim=2) > 1))
        
            c_cc_num += torch.sum(torch.logical_and((torch.sum(states*cc_mask, dim=2) > 1)[:,i],
                                                    (torch.sum(states*coop_mask, dim=2) > 0)[:,i+1]))
            c_cd_num += torch.sum(torch.logical_and((torch.sum(states*cd_mask, dim=2) > 1)[:,i],
                                                    (torch.sum(states*coop_mask, dim=2) > 0)[:,i+1]))
            c_dc_num += torch.sum(torch.logical_and((torch.sum(states*dc_mask, dim=2) > 1)[:,i],
                                                    (torch.sum(states*coop_mask, dim=2) > 0)[:,i+1]))
            c_dd_num += torch.sum(torch.logical_and((torch.sum(states*dd_mask, dim=2) > 1)[:,i],
                                                    (torch.sum(states*coop_mask, dim=2) > 0)[:,i+1]))
        else:
            start_num += torch.sum((torch.sum(states*start_mask, dim=2) > 3))
            c_start_num += torch.sum(torch.logical_and((torch.sum(states*start_mask, dim=2) > 3)[:,i],
                                                    (torch.sum(states*coop_mask, dim=2) > 0)[:,i+1]))

    c_start_prob = c_start_num/start_num
    c_cc_prob = c_cc_num/cc_num
    c_cd_prob = c_cd_num/cd_num
    c_dc_prob = c_dc_num/dc_num
    c_dd_prob = c_dd_num/dd_num

    return c_start_prob, c_cc_prob, c_cd_prob, c_dc_prob, c_dd_prob
    

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

    def log_wandb_ipd_info(self, 
                           r1, 
                           r2, 
                           ent_1, 
                           ent_2,
                           pg_loss_1,
                           pg_loss_2,
                           val_loss_1,
                           val_loss_2,
                           exp_ent_1,
                           exp_ent_2,
                           exp_r1,
                           exp_r2,
                           exp_loss_1,
                           exp_loss_2,
                           c_a,
                           c_b,
                           d_a,
                           d_b,
                           p_c_s_a,
                           p_c_cc_a,
                           p_c_cd_a,
                           p_c_dc_a,
                           p_c_dd_a,
                           p_c_s_b,
                           p_c_cc_b,
                           p_c_cd_b,
                           p_c_dc_b,
                           p_c_dd_b):
        
        self.cum_steps = self.cum_steps + 1
        self.avg_reward_1.insert(0, r1)
        self.avg_reward_2.insert(0, r2)

        self.avg_reward_1 = self.avg_reward_1[0:self.reward_window]
        self.avg_reward_2 = self.avg_reward_2[0:self.reward_window]

        avg_1 = sum(self.avg_reward_1)/len(self.avg_reward_1)
        avg_2 = sum(self.avg_reward_2)/len(self.avg_reward_2)

        wandb_info = {}
        wandb_info['agent_1_avg_reward'] = avg_1
        wandb_info['agent_2_avg_reward'] = avg_2
        wandb_info['total_avg_reward'] = (avg_1 + avg_2)/2
        wandb_info['entropy_1'] = ent_1
        wandb_info['entropy_2'] = ent_2
        wandb_info['exp_entropy_1'] = exp_ent_1
        wandb_info['exp_entropy_2'] = exp_ent_2
        wandb_info['pg_loss_1'] = pg_loss_1
        wandb_info['pg_loss_2'] = pg_loss_2
        wandb_info['value_loss_1'] = val_loss_1
        wandb_info['value_loss_2'] = val_loss_2
        wandb_info['exp_1_avg_reward'] = exp_r1
        wandb_info['exp_2_avg_reward'] = exp_r2
        wandb_info['exp_loss_1'] = exp_loss_1
        wandb_info['exp_loss_2'] = exp_loss_2
        wandb_info['coop_score_a'] = c_a
        wandb_info['coop_score_b'] = c_b
        wandb_info['defect_score_a'] = d_a
        wandb_info['defect_score_b'] = d_b

        wandb_info['P_a(c|s)'] = p_c_s_a
        wandb_info['P_a(c|cc)'] = p_c_cc_a
        wandb_info['P_a(c|cd)'] = p_c_cd_a
        wandb_info['P_a(c|dc)'] = p_c_dc_a
        wandb_info['P_a(c|dd)'] = p_c_dd_a
        wandb_info['P_b(c|s)'] = p_c_s_b
        wandb_info['P_b(c|cc)'] = p_c_cc_b
        wandb_info['P_b(c|cd)'] = p_c_cd_b
        wandb_info['P_b(c|dc)'] = p_c_dc_b
        wandb_info['P_b(c|dd)'] = p_c_dd_b

        wandb.log(wandb_info)
         

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
                       t_r2=None,
                       a_1=None,
                       a_2=None,
                       uc_1=None,
                       uc_2=None,
                       ud_1=None,
                       ud_2=None,
                       ua_1=None,
                       ua_2=None):
                       
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
        wandb_info['self_score_1'] = a_1
        wandb_info['self_score_2'] = a_2

        wandb_info['unconditioned_c_score_1'] = uc_1
        wandb_info['unconditioned_c_score_2'] = uc_2
        wandb_info['unconditioned_d_score_1'] = ud_1
        wandb_info['unconditioned_d_score_2'] = ud_2
        wandb_info['unconditioned_s_score_1'] = ua_1
        wandb_info['unconditioned_s_score_2'] = ua_2

        wandb.log(wandb_info)
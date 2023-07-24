import gym
import torch

class IPD(gym.Env):

    def __init__(self, device, batch_size):
        self.device = device
        self.n_actions = 2
        self.batch_size = batch_size
        self.payout = torch.Tensor([[-1.,-3.],[0.,-2.]]).repeat(batch_size, 1, 1).to(device)

    def step(self, actions):
        a1, a2 = actions
        obs_1 = torch.cat([a1, a2], dim=1)
        obs_2 = torch.cat([a2, a1], dim=1)

        a1 = a1.reshape(self.batch_size, 1, a1.shape[1])
        a2 = a2.reshape(self.batch_size,a2.shape[1], 1)
        a1 = a1.float()
        a2 = a2.float()

        payout_T = torch.transpose(self.payout, 2, 1) 
        r1 = torch.bmm(torch.bmm(a1, self.payout), a2)
        r2 = torch.bmm(torch.bmm(a1, payout_T), a2)

        return (obs_1.float(), obs_2.float()), (r1, r2), {}

    def reset(self):
        return torch.zeros(self.batch_size, 2 * self.n_actions).to(self.device), {}

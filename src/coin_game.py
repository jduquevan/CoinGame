import torch
import numpy as np

"""
OG COIN GAME (But with bugs fixed, also vectorized).
Coin cannot spawn under any agent
Note that both agents can occupy the same spot on the grid
If both agents collect a coin at the same time, they both get the rewards associated
with the collection. To split the rewards (as if taking an expectation where the 
coin is randomly allocated to one of the agents), use --split_coins
"""
class OGCoinGameGPU:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4

    def __init__(self, max_steps, split_coins, batch_size=1, grid_size=3, device="cpu"):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = 4
        self.step_count = None
        self.device = device
        self.split_coins = split_coins
        self.MOVES = torch.stack([
            torch.LongTensor([0, 1]), # right
            torch.LongTensor([0, -1]), # left
            torch.LongTensor([1, 0]), # down
            torch.LongTensor([-1, 0]), # up
        ], dim=0).to(self.device)

    def reset(self):
        self.step_count = 0
        self.red_coin = torch.randint(2, size=(self.batch_size,)).to(self.device)

        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1)

        coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 2, size=(self.batch_size,)).to(self.device)
        minpos = torch.min(red_pos_flat, blue_pos_flat)
        maxpos = torch.max(red_pos_flat, blue_pos_flat)
        coin_pos_flat[coin_pos_flat >= minpos] += 1
        coin_pos_flat[coin_pos_flat >= maxpos] += 1

        # Regenerate coins when both agents are on the same spot, use a uniform
        # distribution among the 8 other possible spots
        same_agents_pos = (minpos == maxpos)
        coin_pos_flat[same_agents_pos] = torch.randint(self.grid_size * self.grid_size - 1, size=(same_agents_pos.sum(),)).to(self.device) + 1 + minpos[same_agents_pos]

        coin_pos_flat = coin_pos_flat % (self.grid_size * self.grid_size)

        # Test distribution of coins
        # print((minpos == maxpos).sum())
        # for i in range(self.grid_size * self.grid_size):
        #     print(torch.logical_and(minpos == maxpos, (coin_pos_flat == (minpos + i) % (self.grid_size * self.grid_size)) ).sum())
        assert (coin_pos_flat == red_pos_flat).sum() == 0
        assert (coin_pos_flat == blue_pos_flat).sum() == 0

        self.coin_pos = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)
        self.set_distances()

        state = self._generate_state()
        state2 = state.clone()
        state2[:, 0] = state[:, 1]
        state2[:, 1] = state[:, 0]
        state2[:, 2] = state[:, 3]
        state2[:, 3] = state[:, 2]
        observations = [state, state2]
        return observations

    def clone_env(self, game):
        self.step_count = game.step_count
        self.red_coin = game.red_coin.repeat(self.batch_size)
        self.red_dist = game.red_dist.repeat(self.batch_size)
        self.blue_dist = game.blue_dist.repeat(self.batch_size)
        self.red_pos = game.red_pos.repeat(self.batch_size, 1)
        self.blue_pos = game.blue_pos.repeat(self.batch_size, 1)
        self.coin_pos = game.coin_pos.repeat(self.batch_size, 1)

    def clone_env_batch(self, game):
        self.step_count = game.step_count
        self.red_coin = game.red_coin.clone()
        self.red_dist = game.red_dist.clone()
        self.blue_dist = game.blue_dist.clone()
        self.red_pos = game.red_pos.clone()
        self.blue_pos = game.blue_pos.clone()
        self.coin_pos = game.coin_pos.clone()

    def clone_env_inv(self, game):
        self.step_count = game.step_count
        self.red_coin = 1 - game.red_coin.repeat(self.batch_size)
        self.red_dist = game.blue_dist.repeat(self.batch_size)
        self.blue_dist = game.red_dist.repeat(self.batch_size)
        self.red_pos = game.blue_pos.repeat(self.batch_size, 1)
        self.blue_pos = game.red_pos.repeat(self.batch_size, 1)
        self.coin_pos = game.coin_pos.repeat(self.batch_size, 1)

    def _generate_coins(self):
        mask = torch.logical_or(self._same_pos(self.coin_pos, self.blue_pos), self._same_pos(self.coin_pos, self.red_pos))
        self.red_coin = torch.where(mask, 1 - self.red_coin, self.red_coin)

        red_pos_flat = self.red_pos[mask,0] * self.grid_size + self.red_pos[mask, 1]
        blue_pos_flat = self.blue_pos[mask, 0] * self.grid_size + self.blue_pos[mask, 1]
        coin_pos_flat = torch.randint(self.grid_size * self.grid_size - 2, size=(self.batch_size,)).to(self.device)[mask]
        minpos = torch.min(red_pos_flat, blue_pos_flat)
        maxpos = torch.max(red_pos_flat, blue_pos_flat)
        coin_pos_flat[coin_pos_flat >= minpos] += 1
        coin_pos_flat[coin_pos_flat >= maxpos] += 1

        same_agents_pos = (minpos == maxpos)

        # Regenerate coins when both agents are on the same spot, regenerate uniform among the 8 other possible spots
        coin_pos_flat[same_agents_pos] = torch.randint(
            self.grid_size * self.grid_size - 1,
            size=(same_agents_pos.sum(),)).to(self.device) + 1 + minpos[same_agents_pos]

        coin_pos_flat = coin_pos_flat % (self.grid_size * self.grid_size)

        # Test distribution of coins
        # print((minpos == maxpos).sum())
        # for i in range(self.grid_size * self.grid_size):
        #     print(torch.logical_and(minpos == maxpos, (coin_pos_flat == (minpos + i) % (self.grid_size * self.grid_size)) ).sum())

        self.coin_pos[mask] = torch.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size), dim=-1)

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:,0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        # 1 - self.red_coin here in order to have the red coin show up as obs in the second to last, rather than last of the 4 dimensions
        coin_pos_flatter = self.coin_pos[:,0] * self.grid_size + self.coin_pos[:,1] + self.grid_size * self.grid_size * (1-self.red_coin) + 2 * self.grid_size * self.grid_size

        state = torch.zeros((self.batch_size, 4*self.grid_size*self.grid_size)).to(self.device)

        state.scatter_(1, coin_pos_flatter[:,None], 1)
        state = state.view((self.batch_size, 4, self.grid_size*self.grid_size))

        state[:,0].scatter_(1, red_pos_flat[:,None], 1)
        state[:,1].scatter_(1, blue_pos_flat[:,None], 1)

        state = state.view(self.batch_size, 4, self.grid_size, self.grid_size)

        return state

    def set_distances(self):
        red_horiz_dist = torch.min((self.coin_pos[:, 1] - self.red_pos[:, 1]) % self.grid_size,
                                    (self.red_pos[:, 1] - self.coin_pos[:, 1]) % self.grid_size)
        red_vertl_dist = torch.min((self.coin_pos[:, 0] - self.red_pos[:, 0]) % self.grid_size,
                                    (self.red_pos[:, 0] - self.coin_pos[:, 0]) % self.grid_size)
        blue_horiz_dist = torch.min((self.coin_pos[:, 1] - self.blue_pos[:, 1]) % self.grid_size,
                                    (self.blue_pos[:, 1] - self.coin_pos[:, 1]) % self.grid_size)
        blue_vertl_dist = torch.min((self.coin_pos[:, 0] - self.blue_pos[:, 0]) % self.grid_size,
                                    (self.blue_pos[:, 0] - self.coin_pos[:, 0]) % self.grid_size)
        self.red_dist = red_horiz_dist + red_vertl_dist
        self.blue_dist = blue_horiz_dist + blue_vertl_dist

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_matches = self._same_pos(self.red_pos, self.coin_pos)
        red_reward = torch.zeros_like(self.red_coin).float()

        blue_matches = self._same_pos(self.blue_pos, self.coin_pos)
        blue_reward = torch.zeros_like(self.red_coin).float()
        
        red_takes_red = torch.logical_and(red_matches, self.red_coin)
        blue_takes_blue = torch.logical_and(blue_matches, 1 - self.red_coin)
        blue_takes_red = torch.logical_and(blue_matches, self.red_coin)
        red_takes_blue = torch.logical_and(red_matches, 1 - self.red_coin)

        red_reward[red_takes_red] = 1
        blue_reward[blue_takes_blue] = 1
        red_reward[red_takes_blue] = 1
        blue_reward[blue_takes_red] = 1

        red_reward[blue_takes_red] += -2
        blue_reward[red_takes_blue] += -2

        self.red_adv = None
        self.blue_adv = None
        self.red_can_blue = torch.logical_and((self.red_dist==1), 1 - self.red_coin)
        self.blue_can_red = torch.logical_and((self.blue_dist==1), self.red_coin)
        self.red_can_red = torch.logical_and((self.red_dist==1), self.red_coin)
        self.blue_can_blue = torch.logical_and((self.blue_dist==1), 1 - self.red_coin)
        self.red_takes_red = red_takes_red
        self.blue_takes_blue = blue_takes_blue
        self.blue_takes_red = blue_takes_red
        self.red_takes_blue = red_takes_blue

        self.set_distances()

        if self.split_coins:
            both_matches = torch.logical_and(self._same_pos(self.red_pos, self.coin_pos), self._same_pos(self.blue_pos, self.coin_pos))
            red_reward[both_matches] *= 0.5
            blue_reward[both_matches] *= 0.5

        total_rb_matches = torch.logical_and(red_matches, 1 - self.red_coin).float().mean()
        total_br_matches = torch.logical_and(blue_matches, self.red_coin).float().mean()

        total_rr_matches = red_matches.float().mean() - total_rb_matches
        total_bb_matches = blue_matches.float().mean() - total_br_matches

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        state2 = state.clone()
        # Because each agent sees the obs as if they are the "main" or "red" agent.
        # This is to be consistent with the self-centric IPD formulation too.
        state2[:, 0] = state[:, 1]
        state2[:, 1] = state[:, 0]
        state2[:, 2] = state[:, 3]
        state2[:, 3] = state[:, 2]
        observations = [state, state2]
        if self.step_count >= self.max_steps:
            done = torch.ones_like(self.red_coin)
        else:
            done = torch.zeros_like(self.red_coin)

        return observations, reward, done, (
        total_rr_matches, total_rb_matches,
        total_br_matches, total_bb_matches)

    def get_moves_shortest_path_to_coin(self, red_agent_perspective=True):
        # Ties broken arbitrarily, in this case, since I check the vertical distance later
        # priority is given to closing vertical distance (making up or down moves)
        # before horizontal moves
        if red_agent_perspective:
            agent_pos = self.red_pos
        else:
            agent_pos = self.blue_pos
        actions = torch.zeros(self.batch_size) - 1
        # assumes red agent perspective
        horiz_dist_right = (self.coin_pos[:, 1] - agent_pos[:, 1]) % self.grid_size
        horiz_dist_left = (agent_pos[:, 1] - self.coin_pos[:, 1]) % self.grid_size

        vert_dist_down = (self.coin_pos[:, 0] - agent_pos[:,
                                                0]) % self.grid_size
        vert_dist_up = (agent_pos[:, 0] - self.coin_pos[:,
                                             0]) % self.grid_size
        actions[horiz_dist_right < horiz_dist_left] = 0
        actions[horiz_dist_left < horiz_dist_right] = 1
        actions[vert_dist_down < vert_dist_up] = 2
        actions[vert_dist_up < vert_dist_down] = 3
        # Assumes no coin spawns under agent
        assert torch.logical_and(horiz_dist_right == horiz_dist_left, vert_dist_down == vert_dist_up).sum() == 0

        return actions.long()

    def get_moves_away_from_coin(self, moves_towards_coin):
        opposite_moves = torch.zeros_like(moves_towards_coin)
        opposite_moves[moves_towards_coin == 0] = 1
        opposite_moves[moves_towards_coin == 1] = 0
        opposite_moves[moves_towards_coin == 2] = 3
        opposite_moves[moves_towards_coin == 3] = 2
        return opposite_moves

    def get_coop_action(self, red_agent_perspective=True):
        # move toward coin if same colour, away if opposite colour
        # An agent that always does this is considered to 'always cooperate'
        moves_towards_coin = self.get_moves_shortest_path_to_coin(red_agent_perspective=red_agent_perspective).to(self.device)
        moves_away_from_coin = self.get_moves_away_from_coin(moves_towards_coin).to(self.device)
        coop_moves = torch.zeros_like(moves_towards_coin) - 1
        if red_agent_perspective:
            is_my_coin = self.red_coin
        else:
            is_my_coin = 1 - self.red_coin

        coop_moves[is_my_coin == 1] = moves_towards_coin[is_my_coin == 1]
        coop_moves[is_my_coin == 0] = moves_away_from_coin[is_my_coin == 0]
        return coop_moves

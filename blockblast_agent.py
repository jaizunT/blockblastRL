import random
import numpy as np
import torch
import torch.nn as nn

import blockblast_calibration as calibration
import blockblast_status as status
import blockblast_play as play

# Scaffold for a BlockBlast RL agent using PyTorch.


class BlockBlastEnv:
    def __init__(self):
        self.refresh_data()

    def _encode_state(self):
        # self.refresh_data()
        board = np.array(self.board, dtype=np.float32)
        tray_tensors = []
        for tray in self.trays:
            tray = np.array(tray, dtype=np.float32)
            tray_tensors.append(tray)
        return board, tray_tensors

    def reset(self):
        play.click_restart()
        self.refresh_data()
        return self._encode_state()

    def step(self, action):
        tray_index, x, y = action
        block = self.trays[tray_index]
        prev_score = self.score
        prev_board = self.board.copy()
        play.place_block(tray_index, x, y, block)
        while not status.background_stable():
            pass
        self.refresh_data()
        reward = self.score - prev_score
        done = play.check_loss(prev_board, tray_index, x, y, self.trays)
        return self._encode_state(), reward, done, {}
    
    def refresh_data(self):
        data = status.screenshot()
        self.board = status.get_board_state(data)
        self.trays = play.get_blocks(data)
        self.score = status.get_score(data)



class PolicyNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, board_tensor):
        return self.net(board_tensor)


def sample_action(logits, mask=None):
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample()
    return action_idx, dist.log_prob(action_idx)


def action_index_to_tuple(action_idx, grid_size=8, trays=3):
    action_idx = int(action_idx)
    per_tray = grid_size * grid_size
    tray = action_idx // per_tray
    rem = action_idx % per_tray
    x = rem % grid_size
    y = rem // grid_size
    return tray, x, y


def main():
    env = BlockBlastEnv()
    action_dim = 3 * 8 * 8
    policy = PolicyNet(action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Placeholder training loop structure.
    for _ in range(10):
        state = env.reset()
        board, _ = state
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        logits = policy(board_tensor)
        action_idx, logp = sample_action(logits)
        action = action_index_to_tuple(action_idx)
        (next_state, reward, done, _) = env.step(action)
        loss = -logp * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if done:
            break


if __name__ == "__main__":
    main()

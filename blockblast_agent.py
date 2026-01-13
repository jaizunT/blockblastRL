import random
import time
import numpy as np
import torch
import torch.nn as nn

import blockblast_calibration as calibration
import blockblast_status as status
import blockblast_play as play

# Scaffold for a BlockBlast RL agent using PyTorch.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

BOARD_SIZE = 8
TRAY_COUNT = 3
TRAY_PAD_SIZE = 5
DEBUG = True


def debug(msg):
    if DEBUG:
        print(f"[agent] {msg}", flush=True)

class BlockBlastEnv:
    def __init__(self):
        self.refresh_data()
        self.moves_since_last_clear = 0
        self.lines_cleared_last_move = 0
        self.combo_count = 0
        self.start = True

    def _pad_tray(self, tray, size=TRAY_PAD_SIZE):
        if tray is None:
            return np.zeros((size, size), dtype=np.float32)
        tray_arr = np.array(tray, dtype=np.float32)
        h, w = tray_arr.shape
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:h, :w] = tray_arr
        return padded

    def _encode_state(self):
        board = np.array(self.board, dtype=np.float32)
        tray_tensors = [self._pad_tray(tray) for tray in self.trays]
        moves = float(self.moves_since_last_clear)
        lines = float(self.lines_cleared_last_move)
        combo = float(self.combo_count)
        return board, tray_tensors, moves, lines, combo

    def reset(self):
        debug("reset: clicking restart")
        if self.start:
            self.start = False
        else:
            play.click_restart()
        status.time.sleep(2)
        debug("reset: refreshing data")
        self.refresh_data()
        self.moves_since_last_clear = 0
        self.lines_cleared_last_move = 0
        self.combo_count = 0
        return self._encode_state()

    def step(self, action):
        time.sleep(2)  # brief pause before action
        tray_index, x, y = action
        # debug print board
        debug("step: current board")
        if DEBUG: status.print_board(self.board)
        debug(f"step: action tray={tray_index} x={x} y={y}")
        block = self.trays[tray_index]
        prev_score = self.score
        prev_board = self.board.copy()
        prev_combo = self.in_combo
        invalid_move = play.invalid_placement(x, y, block, self.board)

        if invalid_move:
            debug("step: invalid move")
            reward = -5.0
            done = False
            return self._encode_state(), reward, done, {}
        
        debug("step: placing block")
        self.lines_cleared_last_move = play.place_block(self.board,tray_index, x, y, block)
        # increase or reset 'moves since last clear' based on lines cleared
        self.moves_since_last_clear = 0 if self.lines_cleared_last_move > 0 else self.moves_since_last_clear + 1

        # wait for background to stabilize before refreshing data
        wait_start = time.time()
        last_log = wait_start
        while not status.background_stable():
            if DEBUG and time.time() - last_log > 0.5:
                debug(f"step: waiting for background_stable ({time.time() - wait_start:.1f}s)")
                last_log = time.time()
        debug("step: background stable, refreshing data")
        self.refresh_data()

        # increase or reset 'combo count' based on lines cleared and previous combo status
        if prev_combo:
            # if not in combo now, reset combo count
            if not self.in_combo:
                self.combo_count = 0
            # elif still in combo, increase combo count by lines cleared
            elif self.lines_cleared_last_move > 0:
                self.combo_count = self.combo_count + self.lines_cleared_last_move
            # else stay the same
        else:
            # if was not in combo previously, start new combo if lines cleared and in combo now
            if self.lines_cleared_last_move > 0 and self.in_combo:
                self.combo_count = self.lines_cleared_last_move
            # else stay the same (should be 0)
            else:
                self.combo_count = 0

        reward = self.score - prev_score
        done = play.check_loss(prev_board, block, x, y, self.trays)
        debug(f"step: reward={reward} done={done} lines_cleared={self.lines_cleared_last_move}")
        return self._encode_state(), reward, done, {}
    
    def refresh_data(self):
        debug("refresh_data: screenshot")
        data = status.screenshot()
        debug("refresh_data: board/trays/score/combo")
        self.board = status.get_board_state(data)
        self.trays = play.get_blocks(data)
        self.score = status.get_score(data)
        self.in_combo = status.is_in_combo(data)


# Need to include tray info and combo status and potentially moves since last line cleared for full state representation.
# Possibly include Lines cleared last move
class PolicyNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        input_dim = (BOARD_SIZE * BOARD_SIZE) + (TRAY_COUNT * TRAY_PAD_SIZE * TRAY_PAD_SIZE) + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, board_tensor, tray_tensors, moves_since_last_clear, lines_cleared_last_move, combo_count):
        board_flat = board_tensor.flatten(start_dim=1)
        trays_flat = tray_tensors.flatten(start_dim=1)
        scalars = torch.stack(
            [moves_since_last_clear, lines_cleared_last_move, combo_count], dim=1
        )
        features = torch.cat([board_flat, trays_flat, scalars], dim=1)
        return self.net(features)


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

# Logs data to moves.txt for analysis
def log_action_to_file(tray_tensors, board, action, moves_since_last_clear, lines_cleared_last_move, combo_count):
    tray_index, x, y = action
    with open("moves.txt", "a") as f:
        # Writes board state
        f.write("Board:\n")
        for row in board:
            f.write("".join(['#' if cell else '.' for cell in row]) + "\n")
        # Writes tray_tensor state
        f.write("Trays:\n")
        for i, tray in enumerate(tray_tensors):
            f.write(f"Tray {i}:\n")
            if tray is None:
                f.write("(empty)\n")
                continue
            for row in tray:
                f.write("".join(['#' if cell else '.' for cell in row]) + "\n")
        # Writes scalar state
        f.write(f"Moves since last clear: {moves_since_last_clear}\n")
        f.write(f"Lines cleared last move: {lines_cleared_last_move}\n")
        f.write(f"Combo count: {combo_count}\n")
        # Writes action taken
        f.write(f"Action: Tray {tray_index} at ({x}, {y})\n")
        f.write("\n")
    

def main():
    print("BlockBlast RL Agent Starting...")
    env = BlockBlastEnv()
    action_dim = 3 * 8 * 8
    policy = PolicyNet(action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    print("Starting training loop...")
    # Placeholder training loop structure.
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            board, trays, moves, lines, combo = state
            board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
            trays_tensor = torch.tensor(np.stack(trays), dtype=torch.float32).unsqueeze(0)
            moves_tensor = torch.tensor([moves], dtype=torch.float32)
            lines_tensor = torch.tensor([lines], dtype=torch.float32)
            combo_tensor = torch.tensor([combo], dtype=torch.float32)
            logits = policy(
                board_tensor, trays_tensor, moves_tensor, lines_tensor, combo_tensor
            )
            action_idx, logp = sample_action(logits)
            action = action_index_to_tuple(action_idx)
            (next_state, reward, done, _) = env.step(action)
            loss = -logp * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state


if __name__ == "__main__":
    main()

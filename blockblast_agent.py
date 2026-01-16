import json
import random
import time
import numpy as np
import torch
import torch.nn as nn

import blockblast_calibration as calibration
import blockblast_status as status
import blockblast_play as play
import blockblast_auto_calibrate as auto

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
        self.game = 0

        self.steps = 0

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
        if h > size or w > size:
            debug(f"pad_tray: cropping tray {h}x{w} to {size}x{size}")
            tray_arr = tray_arr[:size, :size]
            h, w = tray_arr.shape
        padded[:h, :w] = tray_arr
        return padded

    def log_batch(self, board, trays, path="batch_log.jsonl"):
        if any(tray is None for tray in trays):
            return
        record = {
            "board": np.array(board, dtype=int).tolist(),
            "trays": [self._pad_tray(tray).astype(int).tolist() for tray in trays],
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._log_unique_blocks(trays)

    def _serialize_block(self, block):
        block_arr = np.array(block, dtype=int)
        rows = ["".join(str(int(cell)) for cell in row) for row in block_arr]
        return f"{block_arr.shape[0]}x{block_arr.shape[1]}|" + "/".join(rows)

    def _log_unique_blocks(self, trays, path="unique_blocks.txt"):
        existing = set()
        try:
            with open(path, "r") as f:
                for line in f:
                    existing.add(line.strip())
        except FileNotFoundError:
            pass

        new_lines = []
        for tray in trays:
            if tray is None:
                continue
            serialized = self._serialize_block(tray)
            if serialized not in existing:
                existing.add(serialized)
                new_lines.append(serialized)

        if new_lines:
            with open(path, "a") as f:
                for line in new_lines:
                    f.write(line + "\n")

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
            play.click_settings_replay()
        else:
            play.click_restart()
        debug("reset: refreshing data")
        self.refresh_data()
        self.moves_since_last_clear = 0
        self.lines_cleared_last_move = 0
        self.combo_count = 0
        self.steps = 0
        self.game += 1
        return self._encode_state()

    def step(self, action):
        debug(f"Current game: {self.game}")
        debug(f"Current score: {self.score}")
        debug(f"Current combo: {self.in_combo}")
        debug(f"Current combo count: {self.combo_count}")


        tray_index, row, col = action
        reward = 0

        self.steps += 1

        # debug print board
        debug("step: current board")
        if DEBUG: status.print_board(self.board)
        debug(f"step: action tray={tray_index} row={row} col={col}")

        block = self.trays[tray_index]
        pre_state = self._encode_state()

        if DEBUG:
            debug(f"step: tray")
            status.print_all_trays(self.trays)
        tray_screenshot = self.tray_screenshot    

        # If tray is empty, negative reward
        if block is None:
            debug("step: empty tray selected, invalid move")
            reward = -40.0
            done = False
            return self._encode_state(), reward, done, {}
        
        prev_score = self.score
        prev_board = self.board.copy()
        prev_combo = self.in_combo

        self.log_batch(self.board, self.trays)

        # If block not calibrated, raise exception
        class_name = f"{block.shape[0]}x{block.shape[1]}"
        if not auto.is_class_calibrated(class_name, tray_index):
            debug(f"step: block not calibrated for tray {tray_index} class {class_name}")
            status.print_block(block)
            raise Exception("Block not calibrated, please run auto-calibration script.")
        
        invalid_move = play.invalid_placement(col, row, block, self.board)

        if invalid_move:
            debug("step: overlaps / out of bounds, invalid move")
            reward = -20.0
            done = False
            return self._encode_state(), reward, done, {}
        
        # time.sleep(0.5) # brief pause before placement

        debug("step: placing block")
        self.lines_cleared_last_move = play.place_block(self.board, tray_index, col, row, block)
        
        # increase or reset 'moves since last clear' based on lines cleared
        self.moves_since_last_clear = 0 if self.lines_cleared_last_move > 0 else self.moves_since_last_clear + 1

        # wait for background to stabilize before refreshing data if only one tray left that is not None
        if sum(1 for b in self.trays if b is not None) == 1:
            debug("step: batch resetting, waiting longer for stability")
            time.sleep(0.3)
        if self.lines_cleared_last_move > 0:
            debug("step: lines cleared, waiting longer for stability")
            time.sleep(0.5)
        if not invalid_move: 
            debug("step: block placed, waiting until tray clears for stability")
            time.sleep(0.3) # make sure read tray isn't messed up
        while not status.background_stable():
            time.sleep(0.05)
        debug("step: background stable, refreshing data")
        self.refresh_data()
        
        
        score_increase = np.abs(self.score - prev_score)

        standard_loss = play.check_loss(prev_board, block, col, row, self.trays, debug=DEBUG)

        if standard_loss:
            debug("step: loss detected with standard check, no valid moves left")
            debug(f"step: score increase before loss: {score_increase}")
            reward =  - 10.0
            done = True
            debug(f"step: reward={reward} done={done} lines_cleared={self.lines_cleared_last_move}")
            return pre_state, reward, done, {}

        
        curr = time.time()
        while not play.placed_correctly(col, row, block, self.board, prev_board):
            time.sleep(0.05)
            self.refresh_data()
            if time.time() - curr > 5.0:
                if play.video_ad_detected(snapshot=status.screenshot()):
                    debug("step: video ad detected during placement verification, clicking out of ad")
                    play.click_out_of_ad()
                    debug(f"step: score increase before loss: {score_increase}")
                    reward =  - 10.0
                    done = True
                    debug(f"step: reward={reward} done={done} lines_cleared={self.lines_cleared_last_move}")
                    return pre_state, reward, done, {}
                else:
                    debug("step: placement verification timeout!")
                    print("Expected board after placement:")
                    expected = play.expected_board_after_placement(col, row, block, prev_board)
                    status.print_board(expected)
                    print("Actual board after placement:")
                    status.print_board(self.board)

                    # Save tray screenshot for debugging
                    tray_path = f"tray_debug_game{self.game}.png"
                    status.save_screenshot(tray_screenshot, tray_path)
                    debug(f"step: saved tray screenshot to {tray_path}")

                    raise RuntimeError("Placement verification timeout")
            
        debug("step: placed correctly")

        ad_loss = get_valid_move_mask(self.board, self.trays, debug_mask=DEBUG).sum() == 0
        if ad_loss:
            debug("step: ad loss detected with valid move mask, no valid moves left")
            play.click_out_of_ad()
            debug(f"step: score increase before loss: {score_increase}")
            reward =  - 10.0
            done = True
            debug(f"step: reward={reward} done={done} lines_cleared={self.lines_cleared_last_move}")
            return pre_state, reward, done, {}

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

        debug(f"step: score increase: {score_increase}")
        reward = (
            score_increase / 1000.0
            + (self.lines_cleared_last_move**2)
            + (self.combo_count * 0.5) 
            + (0.2 if not prev_combo and self.in_combo else 0.0)
            - (1.0 if prev_combo and self.combo_count == 0 else 0.0) 
            + 0.1 * self.steps
            )
        done = False
        debug(f"step: reward={reward} done={done} lines_cleared={self.lines_cleared_last_move}")
        return self._encode_state(), reward, done, {}
    
    def refresh_data(self):
        debug("refresh_data: screenshot")
        data = status.screenshot()
        self.board = status.get_board_state(data)
        self.trays = play.get_blocks(data)
        self.score = status.get_score(data)
        self.in_combo = status.is_in_combo(data)

        self.tray_screenshot = status.get_trays_screenshot(data)


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
    col = rem % grid_size
    row = rem // grid_size
    return tray, row, col

# Logs data to moves.txt for analysis
def log_action_to_file(tray_tensors, board, action, moves_since_last_clear, lines_cleared_last_move, combo_count):
    tray_index, row, col = action
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
        f.write(f"Action: Tray {tray_index} at (row={row}, col={col})\n")
        f.write("\n")
    

def main():
    print("BlockBlast RL Agent Starting...")
    env = BlockBlastEnv()
    action_dim = 3 * 8 * 8
    policy = PolicyNet(action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Load existing weights if available
    ckpt = torch.load("weights/policy_checkpoint1500.pt", map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])

    save_every = 500
    step_count = 1500

    print("Starting training loop...")
    # Placeholder training loop structure.
    for _ in range(200):  # Number of episodes
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
            valid_move_mask = get_valid_move_mask(env.board, env.trays)
            action_idx, logp = sample_action(logits, mask=valid_move_mask)
            action = action_index_to_tuple(action_idx)
            (next_state, reward, done, _) = env.step(action)
            loss = -logp * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count % save_every == 0:
                torch.save(
                    {"policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
                    f"weights/policy_checkpoint{step_count}.pt",
                )

            state = next_state

def get_valid_move_mask(board, trays, debug_mask=False):
    board_np = np.array(board)
    mask = torch.zeros((1, 3 * 8 * 8), dtype=torch.bool)
    for tray_index in range(3):
        tray = trays[tray_index]
        if tray is None:
            # Explicitly keep all actions for empty trays masked out.
            mask[0, tray_index * 64 : (tray_index + 1) * 64] = False
            continue
        h, w = tray.shape
        for row in range(8 - h + 1):
            for col in range(8 - w + 1):
                if not play.invalid_placement(col, row, tray, board_np):
                    action_idx = tray_index * 64 + row * 8 + col
                    mask[0, action_idx] = True
    if debug_mask and int(mask.sum().item()) == 0:
        count = int(mask.sum().item())
        debug(f"mask: valid actions={count}")
        if count == 0:
            debug("mask: no valid actions; trays:")
            for i, tray in enumerate(trays):
                if tray is None:
                    debug(f"mask: tray {i}=None")
                else:
                    debug(f"mask: tray {i} shape={tray.shape}")
                    if DEBUG:
                        status.print_block(tray)
    return mask



if __name__ == "__main__":
    main()

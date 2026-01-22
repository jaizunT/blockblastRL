import itertools
from dataclasses import dataclass

import numpy as np
import torch

from block_generation_net import BlockGenerationNet

BOARD_SIZE = 8
BATCH_SIZE = 3


def load_unique_blocks(path="unique_blocks.txt"):
    blocks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dims, pattern = line.split("|", 1)
            rows, cols = (int(v) for v in dims.split("x"))
            rows_data = pattern.split("/")
            block = np.array([[int(c) for c in row] for row in rows_data], dtype=int)
            if block.shape != (rows, cols):
                raise ValueError(f"Bad block shape in {path}: {line}")
            blocks.append(block)
    if not blocks:
        raise ValueError(f"No blocks loaded from {path}")
    return blocks


def valid_placement(board, block, row, col):
    block_h, block_w = block.shape
    if row < 0 or col < 0 or row + block_h > BOARD_SIZE or col + block_w > BOARD_SIZE:
        return False
    for r in range(block_h):
        for c in range(block_w):
            if block[r, c] == 1 and board[row + r, col + c] != 0:
                return False
    return True


def apply_placement(board, block, row, col):
    new_board = board.copy()
    block_h, block_w = block.shape
    for r in range(block_h):
        for c in range(block_w):
            if block[r, c] == 1:
                new_board[row + r, col + c] = 1

    rows_full = [r for r in range(BOARD_SIZE) if np.all(new_board[r, :] != 0)]
    cols_full = [c for c in range(BOARD_SIZE) if np.all(new_board[:, c] != 0)]
    if rows_full:
        new_board[rows_full, :] = 0
    if cols_full:
        new_board[:, cols_full] = 0
    lines_cleared = len(rows_full) + len(cols_full)
    return new_board, lines_cleared


def batch_has_solution(board, blocks):
    for perm in itertools.permutations(range(BATCH_SIZE)):
        if _solve_batch(board, [blocks[i] for i in perm], 0):
            return True
    return False


def _solve_batch(board, blocks, idx):
    if idx >= len(blocks):
        return True
    block = blocks[idx]
    block_h, block_w = block.shape
    for row in range(BOARD_SIZE - block_h + 1):
        for col in range(BOARD_SIZE - block_w + 1):
            if not valid_placement(board, block, row, col):
                continue
            next_board, _ = apply_placement(board, block, row, col)
            if _solve_batch(next_board, blocks, idx + 1):
                return True
    return False


@dataclass
class StepResult:
    board: np.ndarray
    batch: list
    reward: float
    done: bool
    info: dict


class BlockBlastSim:
    def __init__(
        self,
        model_path="block_generator/block_generation_model.pth",
        unique_blocks_path="unique_blocks.txt",
        seed=0,
        temperature=1.0,
        max_batch_attempts=200,
        device=None,
    ):
        self.blocks = load_unique_blocks(unique_blocks_path)
        self.num_blocks = len(self.blocks)
        self.temperature = temperature
        self.max_batch_attempts = max_batch_attempts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlockGenerationNet(self.num_blocks).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.reset()

    def reset(self, board=None):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int) if board is None else board.copy()
        self.score = 0
        self.moves_since_last_clear = 999
        self.clear_streak = 0
        self.combo_count = 0
        self.in_combo = False
        self.batch = []
        self.batch_indices = []
        self.batch_used = [False] * BATCH_SIZE
        self._generate_solvable_batch()
        return self._get_state()

    def _get_state(self):
        return {
            "board": self.board.copy(),
            "batch": [b.copy() for b in self.batch],
            "score": self.score,
            "in_combo": self.in_combo,
            "combo_count": self.combo_count,
            "moves_since_last_clear": self.moves_since_last_clear,
        }

    def _sample_batch_indices(self):
        board_tensor = torch.tensor(
            self.board, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            x = board_tensor.view(-1, 1, BOARD_SIZE, BOARD_SIZE)
            x = self.model.conv(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.model.fc1(x))
            h = self.model.init_h(x)
            prev = torch.full(
                (1,),
                self.model.start_token,
                dtype=torch.long,
                device=self.device,
            )
            indices = []
            for _ in range(BATCH_SIZE):
                emb = self.model.block_embed(prev)
                h = self.model.rnn(emb, h)
                logits = self.model.out(h)
                if self.temperature <= 0:
                    idx = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits / self.temperature, dim=-1)
                    idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                indices.append(int(idx.item()))
                prev = idx
        return indices

    def _generate_solvable_batch(self):
        for _ in range(self.max_batch_attempts):
            indices = self._sample_batch_indices()
            blocks = [self.blocks[i] for i in indices]
            if batch_has_solution(self.board, blocks):
                self.batch = blocks
                self.batch_indices = indices
                self.batch_used = [False] * BATCH_SIZE
                return
        raise RuntimeError("Failed to generate a solvable batch.")

    def _update_combo(self, lines_cleared):
        if lines_cleared > 0:
            if self.moves_since_last_clear <= 3:
                self.clear_streak += lines_cleared
            else:
                self.clear_streak = lines_cleared
            self.moves_since_last_clear = 0
        else:
            self.moves_since_last_clear += 1
            if self.moves_since_last_clear > 3:
                self.clear_streak = 0
                self.combo_count = 0
                self.in_combo = False
                return

        if self.clear_streak >= 3:
            self.in_combo = True
            self.combo_count += lines_cleared
        else:
            self.in_combo = False
            self.combo_count = 0

    def _score_move(self, block, lines_cleared):
        # block_cells = int(np.count_nonzero(block))
        # base = block_cells + (lines_cleared * 10)
        # if self.in_combo:
        #     base = int(base * (1.0 + 0.5 * self.combo_count))
        # return base
        return 0

    def _remaining_blocks(self):
        return [b for b, used in zip(self.batch, self.batch_used) if not used]

    def _has_valid_moves(self):
        for block in self._remaining_blocks():
            block_h, block_w = block.shape
            for row in range(BOARD_SIZE - block_h + 1):
                for col in range(BOARD_SIZE - block_w + 1):
                    if valid_placement(self.board, block, row, col):
                        return True
        return False

    def step(self, action):
        tray_index, row, col = action
        if tray_index < 0 or tray_index >= BATCH_SIZE:
            raise ValueError("tray_index out of range")
        if self.batch_used[tray_index]:
            return StepResult(self.board.copy(), self.batch, -5.0, False, {"invalid": "tray_used"})

        block = self.batch[tray_index]
        if not valid_placement(self.board, block, row, col):
            return StepResult(self.board.copy(), self.batch, -5.0, False, {"invalid": "placement"})

        self.board, lines_cleared = apply_placement(self.board, block, row, col)
        self.batch_used[tray_index] = True

        self._update_combo(lines_cleared)
        reward = float(self._score_move(block, lines_cleared))
        self.score += reward

        if all(self.batch_used):
            self._generate_solvable_batch()

        done = not self._has_valid_moves()
        info = {
            "lines_cleared": lines_cleared,
            "combo_count": self.combo_count,
            "in_combo": self.in_combo,
            "score": self.score,
        }
        return StepResult(self.board.copy(), self.batch, reward, done, info)


if __name__ == "__main__":
    sim = BlockBlastSim()
    state = sim.reset()
    print("Initial state score:", state["score"])

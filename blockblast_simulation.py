import itertools
from dataclasses import dataclass

import numpy as np
import torch

from block_generation_net import BlockGenerationNet

BOARD_SIZE = 8
BATCH_SIZE = 3


def calculate_clutter(board):
    total = board.size
    filled = int(np.count_nonzero(board))
    return filled / total if total else 0.0


def calculate_holes(board):
    rows, cols = board.shape
    empty = (board == 0)
    seen = np.zeros_like(empty, dtype=bool)
    queue = []
    for c in range(cols):
        if empty[0, c]:
            seen[0, c] = True
            queue.append((0, c))
        if empty[rows - 1, c]:
            seen[rows - 1, c] = True
            queue.append((rows - 1, c))
    for r in range(rows):
        if empty[r, 0]:
            seen[r, 0] = True
            queue.append((r, 0))
        if empty[r, cols - 1]:
            seen[r, cols - 1] = True
            queue.append((r, cols - 1))
    while queue:
        r, c = queue.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return int(np.sum(empty & ~seen))


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


def sample_solvable_batch_uniform(board, blocks, rng, max_attempts):
    for _ in range(max_attempts):
        indices = rng.integers(0, len(blocks), size=BATCH_SIZE).tolist()
        batch = [blocks[i] for i in indices]
        if batch_has_solution(board, batch):
            return batch, indices
    return None, None


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
        max_batch_attempts=400,
        device=None,
        use_model_batch=False,
    ):
        self.blocks = load_unique_blocks(unique_blocks_path)
        self.num_blocks = len(self.blocks)
        self.temperature = temperature
        self.max_batch_attempts = max_batch_attempts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_model_batch = use_model_batch
        self.model = None
        if self.use_model_batch:
            self.model = BlockGenerationNet(self.num_blocks).to(self.device)
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.reset()

    def reset(self, board=None):
        if board is None:
            sampled_clutter_prob = 0.8
            sampled_min_clutter = float(
                np.clip(self.rng.normal(loc=0.6, scale=0.1), 0.3, 0.95)
            )
            if self.rng.random() < sampled_clutter_prob:
                board = self.generate_clutterered_board(min_clutter=sampled_min_clutter)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int) if board is None else board.copy()
        self.score = 0
        self.steps = 0
        self.moves_since_last_clear = 999
        self.clear_streak = 0
        self.combo_count = 0
        self.in_combo = False
        self.lines_cleared_last_move = 0
        self._recompute_metrics()
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
            "lines_cleared_last_move": self.lines_cleared_last_move,
            "clutter": self.clutter,
            "holes": self.holes,
        }

    def generate_clutterered_board(self, min_clutter=0.4):
        # Generate a random board with at least min_clutter filled using randomly placed 'growing' blocks of size 1 to 6
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        target_clutter = float(min_clutter)
        max_block_size = 6
        max_attempts = 2000
        attempts = 0

        while calculate_clutter(board) < target_clutter and attempts < max_attempts:
            attempts += 1
            empty_cells = np.argwhere(board == 0)
            if empty_cells.size == 0:
                break

            start_idx = int(self.rng.integers(0, len(empty_cells)))
            start = tuple(empty_cells[start_idx])
            desired_size = int(
                np.clip(
                    np.rint(self.rng.normal(loc=3.0, scale=np.sqrt(2.5))),
                    1,
                    max_block_size,
                )
            )

            cells = {start}
            growth_attempts = 0
            max_growth_attempts = desired_size * 10
            while len(cells) < desired_size and growth_attempts < max_growth_attempts:
                growth_attempts += 1
                base = tuple(list(cells)[int(self.rng.integers(0, len(cells)))])
                r, c = base
                neighbors = [
                    (r + dr, c + dc)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= r + dr < BOARD_SIZE
                    and 0 <= c + dc < BOARD_SIZE
                    and board[r + dr, c + dc] == 0
                    and (r + dr, c + dc) not in cells
                ]
                if neighbors:
                    nxt = neighbors[int(self.rng.integers(0, len(neighbors)))]
                    cells.add(nxt)

            for r, c in cells:
                board[r, c] = 1

        return board

    def _sample_batch_indices(self):
        if self.model is None:
            raise RuntimeError("Model batch sampling requested but model is not loaded.")
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

    def _sample_batch_indices_uniform(self):
        return self.rng.integers(0, self.num_blocks, size=BATCH_SIZE).tolist()

    def _generate_solvable_batch(self):
        if self.use_model_batch and self.rng.random() < 0.5:
            for _ in range(self.max_batch_attempts):
                indices = self._sample_batch_indices()
                blocks = [self.blocks[i] for i in indices]
                if batch_has_solution(self.board, blocks):
                    self.batch = blocks
                    self.batch_indices = indices
                    self.batch_used = [False] * BATCH_SIZE
                    return
        batch, indices = sample_solvable_batch_uniform(
            self.board, self.blocks, self.rng, self.max_batch_attempts
        )
        if batch is None:
            raise RuntimeError("Failed to generate a solvable batch.")
        self.batch = batch
        self.batch_indices = indices
        self.batch_used = [False] * BATCH_SIZE

    def _recompute_metrics(self):
        self.clutter = calculate_clutter(self.board)
        self.holes = calculate_holes(self.board)

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

        self.steps += 1
        prev_score = self.score
        prev_combo = self.in_combo
        prev_holes = self.holes
        prev_clutter = self.clutter

        block = self.batch[tray_index]
        if not valid_placement(self.board, block, row, col):
            return StepResult(self.board.copy(), self.batch, -5.0, False, {"invalid": "placement"})

        self.board, lines_cleared = apply_placement(self.board, block, row, col)
        self.lines_cleared_last_move = lines_cleared
        self.batch_used[tray_index] = True

        self._update_combo(lines_cleared)
        self._recompute_metrics()
        score_increase = float(self._score_move(block, lines_cleared))
        self.score += score_increase

        if all(self.batch_used):
            self._generate_solvable_batch()

        clutter_score = -0.5 if (self.clutter > 0.65 and prev_clutter <= 0.65) else 0.0
        delta_holes = prev_holes - self.holes
        survival_bonus = 0.02
        reward = (
            score_increase / 1000.0
            + (self.lines_cleared_last_move**1.5)
            + (self.combo_count * 0.5)
            + (0.2 if not prev_combo and self.in_combo else 0.0)
            - (1.0 if prev_combo and self.combo_count == 0 else 0.0)
            + 0.2 * (self.steps ** 0.5)
            - clutter_score
            + (delta_holes * 0.2)
            + survival_bonus
        )

        done = not self._has_valid_moves()
        if done:
            # Linear terminal reward centered at 18 steps.
            # steps=18 -> 0, below 18 negative, above 18 positive; clip to keep scale comparable.
            reward = float(np.clip((self.steps - 18) * (10.0 / 18.0), -10.0, 10.0))
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

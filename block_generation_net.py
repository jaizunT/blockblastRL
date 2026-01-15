import json
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple feedforward neural network for block generation
class BlockGenerationNet(nn.Module):
    def __init__(self, num_blocks, hidden_size=128):
        super(BlockGenerationNet, self).__init__()
        self.num_blocks = num_blocks
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(32 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3 * num_blocks)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits.view(-1, 3, self.num_blocks)
    

# Grabs board and trays from batch.jsonl and maps the trays to index format based on unique_blocks.txt


def _serialize_block(block):
    block_arr = np.array(block, dtype=int)
    rows = ["".join(str(int(cell)) for cell in row) for row in block_arr]
    return f"{block_arr.shape[0]}x{block_arr.shape[1]}|" + "/".join(rows)


def _trim_block(padded):
    block_arr = np.array(padded, dtype=int)
    rows = np.where(block_arr.any(axis=1))[0]
    cols = np.where(block_arr.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return block_arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def load_batches_with_tray_indices(batch_path="batch_log.jsonl", unique_path="unique_blocks.txt"):
    with open(unique_path, "r") as f:
        unique_lines = [line.strip() for line in f if line.strip()]
    unique_map = {line: idx for idx, line in enumerate(unique_lines)}

    batches = []
    with open(batch_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            board = np.array(record["board"], dtype=int)
            trays = [np.array(tray, dtype=int) for tray in record["trays"]]
            tray_indices = []
            for tray in trays:
                trimmed = _trim_block(tray)
                if trimmed is None:
                    tray_indices.append(None)
                    continue
                serialized = _serialize_block(trimmed)
                tray_indices.append(unique_map.get(serialized))
            batches.append(
                {
                    "board": board,
                    "trays": trays,
                    "tray_indices": tray_indices,
                }
            )
    return batches


def load_num_blocks(unique_path="unique_blocks.txt"):
    with open(unique_path, "r") as f:
        return sum(1 for line in f if line.strip())


def train_block_generation(
    batch_path="batch_log.jsonl",
    unique_path="unique_blocks.txt",
    epochs=5,
    lr=1e-3,
    device="cpu",
):
    batches = load_batches_with_tray_indices(batch_path, unique_path)
    num_blocks = load_num_blocks(unique_path)
    model = BlockGenerationNet(num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in batches:
            board = torch.tensor(batch["board"], dtype=torch.float32, device=device).unsqueeze(0)
            targets = batch["tray_indices"]
            if any(t is None for t in targets):
                continue
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            logits = model(board)
            loss = 0.0
            for tray_idx in range(3):
                loss = loss + criterion(logits[0, tray_idx].unsqueeze(0), targets[tray_idx].unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch + 1}/{epochs} loss={total_loss:.4f}")

    return model

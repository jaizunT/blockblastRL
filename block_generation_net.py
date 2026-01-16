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
            has_unknown = False
            for tray in trays:
                trimmed = _trim_block(tray)
                if trimmed is None:
                    tray_indices.append(None)
                    continue
                serialized = _serialize_block(trimmed)
                idx = unique_map.get(serialized)
                if idx is None:
                    has_unknown = True
                    break
                tray_indices.append(idx)
            if has_unknown:
                continue
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


def _evaluate_split(model, batches, device="cpu"):
    model.eval()
    total = 0
    correct_per_tray = np.zeros(3, dtype=int)
    exact_match = 0
    with torch.no_grad():
        for batch in batches:
            board = torch.tensor(batch["board"], dtype=torch.float32, device=device).unsqueeze(0)
            targets = batch["tray_indices"]
            if any(t is None for t in targets):
                continue
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            logits = model(board)[0]
            preds = torch.argmax(logits, dim=1)
            total += 1
            for i in range(3):
                if preds[i].item() == targets[i].item():
                    correct_per_tray[i] += 1
            if torch.all(preds == targets):
                exact_match += 1
    if total == 0:
        return {"count": 0, "tray_acc": [0.0, 0.0, 0.0], "exact_acc": 0.0}
    tray_acc = (correct_per_tray / total).tolist()
    exact_acc = exact_match / total
    return {"count": total, "tray_acc": tray_acc, "exact_acc": exact_acc}


def train_block_generation(
    batch_path="batch_log.jsonl",
    unique_path="unique_blocks.txt",
    epochs=5,
    lr=1e-3,
    device="cpu",
    split=(0.8, 0.1, 0.1),
    seed=42,
):
    batches = load_batches_with_tray_indices(batch_path, unique_path)
    if not batches:
        raise ValueError("No training batches found.")
    num_blocks = load_num_blocks(unique_path)
    model = BlockGenerationNet(num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    rng = np.random.default_rng(seed)
    rng.shuffle(batches)
    train_frac, val_frac, test_frac = split
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("split must sum to 1.0")
    n = len(batches)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_batches = batches[:n_train]
    val_batches = batches[n_train:n_train + n_val]
    test_batches = batches[n_train + n_val:]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_batches:
            board = torch.tensor(batch["board"], dtype=torch.float32, device=device).unsqueeze(0)
            targets = torch.tensor(batch["tray_indices"], dtype=torch.long, device=device)
            logits = model(board)
            loss = 0.0
            for tray_idx in range(3):
                loss = loss + criterion(
                    logits[0, tray_idx].unsqueeze(0),
                    targets[tray_idx].unsqueeze(0),
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics = _evaluate_split(model, val_batches, device=device) if val_batches else None
        if val_metrics:
            print(
                f"Epoch {epoch + 1}/{epochs} loss={total_loss:.4f} "
                f"val_exact={val_metrics['exact_acc']:.3f} "
                f"val_tray={val_metrics['tray_acc']}"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} loss={total_loss:.4f}")

    test_metrics = _evaluate_split(model, test_batches, device=device) if test_batches else None
    if test_metrics:
        print(
            f"Test exact={test_metrics['exact_acc']:.3f} "
            f"test_tray={test_metrics['tray_acc']} "
            f"count={test_metrics['count']}"
        )
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_block_generation(
        batch_path="batch_log.jsonl",
        unique_path="unique_blocks.txt",
        epochs=10,
        lr=1e-3,
        device=device,
    )
    torch.save(model.state_dict(), "block_generation/block_generation_model.pth")

if __name__ == "__main__":
    main()

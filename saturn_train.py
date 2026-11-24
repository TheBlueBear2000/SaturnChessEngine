print("Starting...")
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime as time
from pathlib import Path

from network import SaturnNetwork
from util import mask_to_bitmap

BASE = Path(__file__).resolve().parent

### CONFIG ###
DATA_FILE = BASE / "training_positions.bin"
BATCH_SIZE = 32
EPOCHS = 1
SHUFFLE_CHUNK = 100_000  # records held in RAM and shuffled at a time (~10 MB)
FEEDBACK_STEPS = 50  # batches between progress prints / loss samples
CHECKPOINT_FRACTION = 25  # save every 1/25th (4%) of total steps
PARAM_SIZE = (
    13  # 4 castling + 1 turn + 8 en-passant file one-hot  <-- must match SaturnNetwork
)

(BASE / "models").mkdir(exist_ok=True)
(BASE / "plots").mkdir(exist_ok=True)

CE = nn.CrossEntropyLoss()
CE_SUM = nn.CrossEntropyLoss(reduction="sum")

### RECORD LAYOUT (must match the compile script) ###
# "<12QI": 12 x uint64 bitboards + uint32 meta = 100 bytes
RECORD_DTYPE = np.dtype([("bb", "<u8", (12,)), ("meta", "<u4")])
assert RECORD_DTYPE.itemsize == 100

# meta bits: 0 turn | 1-4 castling | 5-8 ep file (8 = none) | 9-23 move (6 from, 6 to, 3 promo)
SHIFTS = np.arange(63, -1, -1, dtype=np.uint64)

# Compile script promo ints: 0 none, 1 q, 2 r, 3 b, 4 n
# Model's promotion head order: q, n, b, r
PROMO_TO_HEAD = np.array([-1, 0, 3, 2, 1], dtype=np.int64)


def unpack_boards(bb):
    """(B, 12) uint64 -> (B, 12, 8, 8) float32 (bit i -> row i//8, col i%8)."""
    bits = (bb[:, :, None] >> SHIFTS) & np.uint64(1)
    return bits.astype(np.float32).reshape(-1, 12, 8, 8)


def unpack_params(meta):
    """(B,) uint32 -> (B, PARAM_SIZE) float32."""
    meta = meta.astype(np.int64)
    white = meta & 1
    castling = (meta >> 1) & 0xF
    ep = (meta >> 5) & 0xF

    cols = [
        (castling >> 3) & 1,
        (castling >> 2) & 1,
        (castling >> 1) & 1,
        castling & 1,
        white,
    ]
    ep_onehot = (ep[:, None] == np.arange(8)[None, :]).astype(
        np.float32
    )  # 8 = none -> all zeros
    return np.concatenate(
        [np.stack(cols, axis=1).astype(np.float32), ep_onehot], axis=1
    )


### LOAD MODEL ###
print("Loading model...")
with open(BASE / "allowed_moves.json", "r") as file:
    allowed_moves = json.load(file)
n_outputs = len(allowed_moves)
model = SaturnNetwork(n_outputs)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# Lookup table: (from | to << 6) -> index in allowed_moves, or -1
move_lookup = np.full(4096, -1, dtype=np.int64)
for idx, mv in enumerate(allowed_moves):
    mv = mv.lower()
    f = (ord(mv[0]) - 97) + (int(mv[1]) - 1) * 8
    t = (ord(mv[2]) - 97) + (int(mv[3]) - 1) * 8
    move_lookup[f | (t << 6)] = idx

### LOAD DATA ###
print("Opening compiled data...")
records = np.memmap(
    DATA_FILE, dtype=RECORD_DTYPE, mode="r"
)  # ignores any trailing partial record
N = len(records)
STEPS_PER_EPOCH = N // BATCH_SIZE
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
print(f"{N} records, {STEPS_PER_EPOCH} steps per epoch")

# Sanity check: vectorised unpacking must agree with the old mask_to_bitmap
try:
    ref = np.array(
        [np.array(mask_to_bitmap(int(c))) for c in records[0]["bb"]], dtype=np.float32
    )
except Exception as e:
    print(f"Could not verify bitmap ordering against mask_to_bitmap: {e}")


def batches():
    """Yield shuffled batches: chunk order and records within each chunk are shuffled."""
    chunk_starts = np.arange(0, N, SHUFFLE_CHUNK)
    np.random.shuffle(chunk_starts)
    for start in chunk_starts:
        chunk = np.array(records[start : start + SHUFFLE_CHUNK])  # load into RAM
        np.random.shuffle(chunk)
        for i in range(0, len(chunk) - BATCH_SIZE + 1, BATCH_SIZE):
            yield chunk[i : i + BATCH_SIZE]


### TRAIN ###
print("Beginning training...")
training_start_time = time.now()
step = 0
records_skipped = 0
losses = []
running_loss = 0.0
running_steps = 0
loss = torch.tensor(0.0)

model.train()
for epoch in range(EPOCHS):
    for batch in batches():
        meta = batch["meta"].astype(np.int64)
        move_key = (meta >> 9) & 0xFFF  # from | to << 6
        promo_int = (meta >> 21) & 0x7

        targets = move_lookup[move_key]
        valid = (targets >= 0) & (
            promo_int <= 4
        )  # promo > 4 can't come from the compile script
        records_skipped += int((~valid).sum())
        if not valid.any():
            continue

        board = torch.from_numpy(unpack_boards(batch["bb"][valid]))
        params = torch.from_numpy(unpack_params(batch["meta"][valid]))
        move_target = torch.from_numpy(targets[valid])
        promo_target = torch.from_numpy(PROMO_TO_HEAD[promo_int[valid]])

        optimizer.zero_grad()
        move_logits, promotion_logits = model(board, params)

        loss_move = CE(move_logits, move_target)

        promo_mask = promo_target >= 0
        if promo_mask.any():
            loss_promotion = CE_SUM(
                promotion_logits[promo_mask], promo_target[promo_mask]
            ) / len(promo_target)
        else:
            loss_promotion = torch.tensor(0.0)

        loss = loss_move + 0.2 * loss_promotion
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1
        running_loss += loss.item()
        running_steps += 1

        if step % FEEDBACK_STEPS == 0:
            td = ((time.now() - training_start_time) / step) * (TOTAL_STEPS - step)
            progress_string = (
                f"Epoch {epoch + 1}/{EPOCHS}, step {step}/{TOTAL_STEPS}, "
                f"ETA: {td.days:02d}d, {td.seconds//3600:02d}:{(td.seconds//60)%60:02d}:{td.seconds%60:02d}, "
                f"Current loss: {loss.item():.4f}{20*' '}"
            )
            print(progress_string, end="\r")
            with open(BASE / "progress.txt", "w") as f:
                f.write(progress_string)
            losses.append(running_loss / running_steps)
            running_loss = 0.0
            running_steps = 0

        # Save a model and loss diagram every 4%
        if step % max(1, TOTAL_STEPS // CHECKPOINT_FRACTION) == 0:
            torch.save(
                model.state_dict(),
                BASE / f"models/checkpoint_{step}_saturn_weights.pth",
            )
            print(f"Saved model at step {step}{50 * ' '}")
            plt.clf()
            plt.plot([FEEDBACK_STEPS * i for i in range(len(losses))], losses)
            plt.savefig(BASE / "plots/losses_plot.png")

print()
time_taken = time.now() - training_start_time
print(
    f"Training complete! Final loss: {loss.item():.4f}. {records_skipped} records skipped (move not in allowed list).\n"
    f"Total time taken: {time_taken.days:02d}d, {time_taken.seconds//3600:02d}:{(time_taken.seconds//60)%60:02d}:{time_taken.seconds%60:02d}\n"
    f"Saving..."
)

torch.save(model.state_dict(), BASE / "models/final_saturn_weights.pth")
plt.clf()
plt.plot([FEEDBACK_STEPS * i for i in range(len(losses))], losses)
plt.savefig(BASE / "plots/losses_plot.png")
print("Saved and done!")

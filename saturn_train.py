print("Starting...")
from datasets import load_dataset
import translator as translator
from network import SaturnNetwork
from util import mask_to_bitmap

import json
import torch
import torch.nn as nn
from datetime import datetime as time
from pathlib import Path
import requests
import duckdb

CE = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

### LOAD MODEL ###
print("Loading model...")
with open(Path(__file__).resolve().parent / "allowed_moves.json", "r") as file:
    allowed_moves = json.load(file)
n_outputs = len(allowed_moves)  # Update to gather automatically from move loading
model = SaturnNetwork(n_outputs)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


### GET DATA ###
print("Gathering data...")
DATASET = "Lichess/chess-position-evaluations"

API_TOKEN = open(Path(__file__).resolve().parent / "hf_key.txt", "r").read()
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = f"https://datasets-server.huggingface.co/size?dataset={DATASET}"
data = requests.get(API_URL, headers=headers).json()
DATASET_SIZE = data["size"]["dataset"]["num_rows"]

LIMIT = 1000000  # If None, whole dataset is streamed
TOTAL_ROWS = DATASET_SIZE if LIMIT == None else LIMIT

streamed_dataset = load_dataset(DATASET, split="train", streaming=True)
positions = streamed_dataset.take(LIMIT) if LIMIT is not None else streamed_dataset

print("Beginning training...")
position_num = 0
positions_failed = 0
positions_skipped = 0
model.train()
startTime = time.now()
for position in positions:
    position_num += 1

    # First each position must be translated into a set of board states for training
    # try:
    #     states = translator.extract_states(position["fen"], position["line"])
    # except Exception as e:
    #     positions_failed += 1
    #     print(f"Error occured within position {position_num}, {position['fen']}:\n{e}")
    #     break
    states = translator.extract_states(position["fen"], position["line"])

    for turn in states:
        optimizer.zero_grad()  # Reset gradients

        board_state = turn["board"]
        board = []
        # Translate bitboard numbers into 3D bitmap
        for channel in board_state:
            board.append(mask_to_bitmap(channel))
        board = torch.tensor(board, dtype=torch.float32)
        board = board.unsqueeze(0)

        params = turn["params"]["castling_rights"] + turn["params"]["scores"]
        params = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        # Get the move's index as its target value
        full_move = turn["move"]
        move = full_move[:4]
        try:
            move_target = allowed_moves.index(move)  # Remove promotion before scanning
        except:
            # If move fails, ignore rest of position
            print(f"\nERROR: Move {move} is not in allowed moves list")
            break

        move_target = torch.tensor([move_target], dtype=torch.long)

        # Extract promotion data from move
        promotion = False
        if len(full_move) == 5:
            promotion_target = ["q", "n", "b", "r"].index(full_move[4].lower())
            promotion = True

        # Train
        move_logits, promotion_logits = model(board, params)

        loss_move = CE(move_logits, move_target)

        if promotion:
            promotion_target = torch.tensor([promotion_target], dtype=torch.long)
            loss_promotion = CE(promotion_logits, promotion_target)
        else:
            loss_promotion = torch.tensor(0.0)

        loss = loss_move + int(promotion) * 0.2 * loss_promotion

        loss.backward()

        optimizer.step()

    if position_num % 20 == 0:
        td = ((time.now() - startTime) / position_num) * (TOTAL_ROWS - position_num)

        print(
            f"Position number {position_num}, ETA: {td.days:02d}d, {td.seconds//3600:02d}:{(td.seconds//60)%60:02d}:{td.seconds%60:02d}, Current loss: {loss.item()}{20*' '}",
            end="\r",
        )

    # Save a model every 10%
    if position_num % (TOTAL_ROWS // 10) == 0:
        torch.save(
            model.state_dict(), f"models/checkpoint_{position_num}_saturn_weights.pth"
        )
        print(f"Saved model at position number {position_num}{50 * ' '}")

print()  # Bypass carridge return of progress listing

print(
    f"Training complete! Finished with final loss of {loss.item()}. {positions_failed} positions failed, {positions_skipped} positions skipped.\nSaving..."
)

torch.save(
    model.state_dict(),
    Path(__file__).resolve().parent / "models/final_saturn_weights.pth",
)

print("Saved and done!")

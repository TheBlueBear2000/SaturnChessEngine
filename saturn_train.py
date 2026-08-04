print("Starting...")
from datasets import load_dataset
import translate_game as translate_game
from network import SaturnNetwork
from util import mask_to_bitmap

import json
import torch
import torch.nn as nn
from datetime import datetime as time

CE = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

### LOAD MODEL ###
print("Loading model...")
with open("allowed_moves.json", "r") as file:
    allowed_moves = json.load(file)
n_outputs = len(allowed_moves)  # Update to gather automatically from move loading
model = SaturnNetwork(n_outputs)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


### GET DATA ###
print("Gathering data...")

LIMIT = None  # If None, whole dataset is streamed
# Looked into getting num rows programatically, but it requires an API key and request
TOTAL_ROWS = 931121 if LIMIT == None else LIMIT

streamed_dataset = load_dataset(
    "Lichess/tournament-chess-games", split="train", streaming=True
)

games = streamed_dataset.take(LIMIT) if LIMIT is not None else streamed_dataset

# Limit to a number
# detailed_dataset = streamed_dataset.take(500)

# Convert the stream subset to a list of games (other information is irrelevant)
# dataset = [game["movetext"] for game in list(detailed_dataset)]

# print(f"Number of games gathered: {len(dataset)}")

print("Beginning training...")
game_num = 0
games_failed = 0
model.train()
startTime = time.now()
for game in games:
    game_num += 1
    # First each game must be translated into a set of board states for training
    movetext = game["movetext"]
    try:
        translated_game = translate_game.translate(movetext)
    except Exception as e:
        games_failed += 1
        print(f"Error occured within game {game_num}, {game['GameURL']}:\n{e}")
        continue

    for turn in translated_game:
        optimizer.zero_grad()  # Reset gradients

        board_state = turn["board"]
        board = []
        # Translate bitboard numbers into 3D bitmap
        for channel in board_state:
            board.append(mask_to_bitmap(channel))
        board = torch.tensor(board, dtype=torch.float32)
        board = board.unsqueeze(0)

        board_params = turn["params"]
        params = board_params["castling_rights"] + board_params["scores"]
        params = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        # Get the move's index as its target value
        full_move = turn["move"]
        move = full_move[:4]
        try:
            move_target = allowed_moves.index(move)  # Remove promotion before scanning
        except:
            # If move fails, ignore rest of game
            print(f"\nMove {move} is not in list ??")
            break

        move_target = torch.tensor([move_target], dtype=torch.long)

        # Extract promotion data from move
        promotion = False
        if len(full_move) == 5:
            promotion_target = ["q", "n", "b", "r"].index(full_move[4])
            promotion = True

        eval_target = turn["eval"]
        eval_target = torch.tensor([[eval_target]], dtype=torch.float32)

        # Train
        move_logits, promotion_logits, eval = model(board, params)

        loss_move = CE(move_logits, move_target)
        loss_eval = MSE(eval, eval_target)

        if promotion:
            promotion_target = torch.tensor([promotion_target], dtype=torch.long)
            loss_promotion = CE(promotion_logits, promotion_target)
        else:
            loss_promotion = torch.tensor(0.0)

        loss = loss_move + int(promotion) * 0.2 * loss_promotion + 0.1 * loss_eval

        loss.backward()

        optimizer.step()

    if game_num % 20 == 0:
        td = ((time.now() - startTime) / game_num) * (TOTAL_ROWS - game_num)

        print(
            f"Game number {game_num}, ETA: {td.days:02d}d, {td.seconds//3600:02d}:{(td.seconds//60)%60:02d}:{td.seconds%60:02d}, Current loss: {loss.item()}{20*' '}",
            end="\r",
        )

    # Save model every 10,000 games
    # if game_num % 10000 == 0:
    if game_num in [
        1,
        10,
        100,
        1000,
        10000,
        50000,
        100000,
        250000,
        500000,
        750000,
        900000,
    ]:
        torch.save(
            model.state_dict(), f"models/checkpoint_{game_num}_saturn_weights.pth"
        )
        print(f"Saved model at game number {game_num}{100 * ' '}")

print()  # Bypass carridge return of progress listing

print(
    f"Training complete! Finished with final loss of {loss.item()}. {games_failed} games failed.\nSaving..."
)

torch.save(model.state_dict(), "models/final_saturn_weights.pth")

print("Saved and done!")

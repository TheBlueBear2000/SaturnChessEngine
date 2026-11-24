print("Starting...")
from datasets import load_dataset
import translator as translator

# from saturn_shared import File

from pathlib import Path
from datetime import datetime as time

import struct
import requests
import traceback

FEEDBACK_STEPS = 50
LIMIT = 1000000  # If None, whole dataset is streamed


# Each 100 bytes is a position by:
# 12x64 = 768 bits of bitboard
# 1 bit for white/black turn
# 4 bits for castling rights
# 3 bits for enpassant file
# 3x4 + 3 = 15 bits for associated move
# total = 791 bits per position
# around 100 bytes

RECORD_FORMAT = "<12QI"
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)  # Exactly 100 bytes

PROMO_TO_INT = {None: 0, "q": 1, "r": 2, "b": 3, "n": 4}
INT_TO_PROMO = {v: k for k, v in PROMO_TO_INT.items()}


def sq_to_int(sq_str):
    file_idx = ord(sq_str[0]) - ord("a")
    rank_idx = int(sq_str[1]) - 1
    if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
        raise ValueError(f"Bad square {sq_str!r}")
    return rank_idx * 8 + file_idx


def int_to_sq(sq_int):
    """Converts 12 -> 'e2'."""
    file_char = chr(ord("a") + (sq_int % 8))
    rank_char = str((sq_int // 8) + 1)
    return file_char + rank_char


def encode_move(move_str):
    """Encodes 'e2e4' or 'f7f8q' into a 15-bit integer."""
    move_str = move_str.lower()
    from_sq = sq_to_int(move_str[0:2])
    to_sq = sq_to_int(move_str[2:4])
    promo_char = move_str[4].lower() if len(move_str) == 5 else None  # Grab 5th char
    promo = PROMO_TO_INT[promo_char]
    return from_sq | (to_sq << 6) | (promo << 12)


def decode_move(move_int):
    """Decodes a 15-bit integer back into 'e2e4' or 'f7f8q'."""
    from_sq = move_int & 0x3F
    to_sq = (move_int >> 6) & 0x3F
    promo = (move_int >> 12) & 0x7
    promo_str = INT_TO_PROMO[promo] or ""
    return int_to_sq(from_sq) + int_to_sq(to_sq) + promo_str


def pack_turn(turn):
    # 1. Extract the 12 bitboards
    bitboards = (
        list(turn["board"].values())
        if isinstance(turn["board"], dict)
        else turn["board"]
    )

    # 2. Pack castling rights
    b3, b2, b1, b0 = turn["params"]["castling_rights"]
    castling = (b3 << 3) | (b2 << 2) | (b1 << 1) | b0

    # 3. Safely encode ep_file (handles None, -1, or 8)
    raw_ep = turn["params"]["previousMovePawnPushedFile"]
    ep_file = 8 if raw_ep in (None, -1, 8) else raw_ep.value

    # 4. Encode move string from turn["move"]
    move_int = encode_move(turn["move"])

    # 5. Combine everything into 32-bit meta
    meta = (
        (int(turn["whiteMove"]) & 0x1)
        | ((castling & 0xF) << 1)
        | ((ep_file & 0xF) << 5)
        | ((move_int & 0x7FFF) << 9)
    )

    return struct.pack(RECORD_FORMAT, *bitboards, meta)


### GET DATA ###
print("Gathering data...")
DATASET = "Lichess/chess-position-evaluations"

API_TOKEN = open(Path(__file__).resolve().parent / "hf_key.txt", "r").read()
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = f"https://datasets-server.huggingface.co/size?dataset={DATASET}"
data = requests.get(API_URL, headers=headers).json()
DATASET_SIZE = data["size"]["dataset"]["num_rows"]

TOTAL_ROWS = DATASET_SIZE if LIMIT == None else LIMIT

streamed_dataset = load_dataset(DATASET, split="train", streaming=True)
positions = streamed_dataset.take(LIMIT) if LIMIT is not None else streamed_dataset

with open("training_positions.bin", "wb") as file:
    position_num = 0
    startTime = time.now()

    for position in positions:
        position_num += 1

        # First each position must be translated into a set of board states for training
        try:
            states = translator.extract_states(position["fen"], position["line"])
        except Exception as e:
            positions_failed += 1
            print(
                f"Error occured within position {position_num}, {position['fen']}:\n{e}\nFull Traceback:\n{traceback.format_exc()}"
            )
            break

        for turn in states:
            data = pack_turn(turn)
            file.write(data)

        if position_num % FEEDBACK_STEPS == 0:
            td = ((time.now() - startTime) / position_num) * (TOTAL_ROWS - position_num)
            progress_string = f"Position number {position_num}, ETA: {td.days:02d}d, {td.seconds//3600:02d}:{(td.seconds//60)%60:02d}:{td.seconds%60:02d}"
            print(
                progress_string,
                end="\r",
            )

print("Saved and done!")

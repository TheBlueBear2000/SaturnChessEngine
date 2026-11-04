import saturn_shared as saturn_shared

import re


def extract_states(fen: str, line: str):
    output = []
    info = translate_fen(fen)
    board = saturn_shared.Board(info["bitboard"], info["metadata"])
    for move in line.split():
        output.append(
            {
                "board": board.getBoardCopy(),
                "params": board.getParams(),
                "move": move.upper(),
            }
        )
        # Update board after recording previous position, so that each move corresponds to previous board state

        try:
            board.update(move.upper())
        except Exception as e:
            print(board.render(), "\n", fen, line, "\n", e)

    return output


def extract_fen(fen):
    pattern = re.compile(
        r"""
        ^([prnbqkPRNBQK1-8/]+)  # Piece placement
        \s+
        ([wb])                  # Active colour
        \s+
        (K?Q?k?q? | -)          # Castling rights
        \s+
        ([a-h][36]|-)           # En passant target
        $
        """,
        re.VERBOSE,
    )

    match = pattern.match(fen.strip())

    if not match:
        raise ValueError("Invalid FEN string")

    return {
        "piece_placement": match.group(1),
        "active_colour": match.group(2),
        "castling": match.group(3),
        "en_passant": match.group(4),
    }


def translate_fen(fen):
    bitboard = [
        0b0000000000000000000000000000000000000000000000000000000000000000,  # White pawns
        0b0000000000000000000000000000000000000000000000000000000000000000,  # White rooks
        0b0000000000000000000000000000000000000000000000000000000000000000,  # White knights
        0b0000000000000000000000000000000000000000000000000000000000000000,  # White bishops
        0b0000000000000000000000000000000000000000000000000000000000000000,  # White queen
        0b0000000000000000000000000000000000000000000000000000000000000000,  # White king
        0b0000000000000000000000000000000000000000000000000000000000000000,  # Black pawns
        0b0000000000000000000000000000000000000000000000000000000000000000,  # Black rooks
        0b0000000000000000000000000000000000000000000000000000000000000000,  # Black knights
        0b0000000000000000000000000000000000000000000000000000000000000000,  # Black bishops
        0b0000000000000000000000000000000000000000000000000000000000000000,  # Black queen
        0b0000000000000000000000000000000000000000000000000000000000000000,  # Black king
    ]

    info = extract_fen(fen)

    rows = info["piece_placement"].split("/")

    PIECES = ["P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k"]

    for y, row in enumerate(rows):
        x = 0
        for char in row:
            if char.isdigit():
                x += int(char)
            else:
                x += 1
                mask = 1 << ((x - 1) + (8 * (7 - y)))
                # print(
                #    f"adding {char} to coordinate ({x}, {y}). Mask is {mask}. bitboard index is {PIECES.index(char)}"
                # )
                bitboard[PIECES.index(char)] |= mask

    castling = (
        "Q" in info["castling"],
        "K" in info["castling"],
        "q" in info["castling"],
        "k" in info["castling"],
    )

    whiteMove = info["active_colour"] == "w"

    previousMovePawnPushedFile = None
    if info["en_passant"] != "-":
        previousMovePawnPushedFile = saturn_shared.File(
            ord(info["en_passant"][0].lower()) - 97
        )

    # print(f"Bitboard: {bitboard}")

    return {
        "bitboard": bitboard,
        "metadata": {
            "castling": castling,
            "previousMovePawnPushedFile": previousMovePawnPushedFile,
            "whiteMove": whiteMove,
        },
    }

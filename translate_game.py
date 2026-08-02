import re
import saturn_shared

PIECE_CHANNELS = [None, "R", "N", "B", "Q", "K"]


def parse_eval(eval):
    if eval.startswith("#"):
        return 100000 - int(eval[1:])
    elif eval.startswith("-#"):
        return -100000 + int(eval[2:])
    else:
        return int(round(float(eval) * 100))


# Takes in a game as a string, produces a set of board states and parameters
def translate(game: str):

    output = []

    # Extract moves
    game_pattern = re.compile(
        r"""
        \d+\.\s+
        (\S+)                     # White move
        \s+\{\s+\[%eval\s+([^\]]+)\]   # White eval
        .*?\}
        \s+\d+\.\.\.\s+
        (\S+)                     # Black move
        \s+\{\s+\[%eval\s+([^\]]+)\]   # Black eval
        .*?\}
        """,
        re.VERBOSE | re.DOTALL,
    )

    raw_moves = []
    evals = []

    for wm, we, bm, be in game_pattern.findall(game):
        raw_moves.append(wm)
        raw_moves.append(bm)
        evals.append(parse_eval(we))
        evals.append(parse_eval(be))

    # Translate moves into server LAN by simulating game
    move_pattern = re.compile(
        r"""
    ^(?:
        (O-O(?:-O)?)                     # Castling
    |
        ([KQRBN])?                       # Piece (empty = pawn)
        ([a-h1-8]{0,2})                  # Disambiguation
        (x)?                             # Capture
        ([a-h][1-8])                     # Destination square
        (?:=([QRBN]))?                   # Promotion
        ([+#])?                          # Check or mate
    )$
    """,
        re.VERBOSE,
    )
    board = saturn_shared.Board()  # Initiate game
    whiteMove = True
    for m_i, move in enumerate(raw_moves):
        # First figure out move full LAN notation based on gamestate, then make move
        move = move.strip("!").strip("?")  # Strip quality annotation
        (
            castling,
            piece,
            disambiguation,
            capture,
            destination,
            promotion,
            checkOrMate,
        ) = move_pattern.match(move).groups()

        capture = capture == "x"  # Turn capture into boolean

        startPos = None

        if castling == "O-O-O":  # Long castle
            if whiteMove:
                startPos = saturn_shared.Position("E", 1)
                endPos = saturn_shared.Position("C", 1)
            else:
                startPos = saturn_shared.Position("E", 8)
                endPos = saturn_shared.Position("C", 8)
        elif castling == "O-O":  # Short castle
            if whiteMove:
                startPos = saturn_shared.Position("E", 1)
                endPos = saturn_shared.Position("G", 1)
            else:
                startPos = saturn_shared.Position("E", 8)
                endPos = saturn_shared.Position("G", 8)

        # If move is not castling, standard info is extracted
        else:
            endPos = saturn_shared.Position(destination[0], destination[1])
            if whiteMove:
                possiblePieces = board.state[:6]
            else:
                possiblePieces = board.state[6:]
            piece_channel = PIECE_CHANNELS.index(piece)
            pieceTypeMask = possiblePieces[piece_channel]
            all_piece_positions = board.getMaskPositions(pieceTypeMask)

            # Filter by disambiguation
            positions = []
            canSee = []
            for pos in all_piece_positions:
                if disambiguation in str(pos).lower():
                    positions.append(pos)

            # If there are still multiple candidates, check which can attack the square depending on type
            if len(positions) > 1:
                canSee = board.posAttackedBy(endPos, whiteMove, board, forTake=capture)
                positions = [posit for posit in positions if posit in canSee]

                if len(positions) == 1:
                    startPos = positions[0]
                else:
                    # If combinational filtering doesnt fix, one piece must be held by threatened check
                    # Best way to find which is to simulate all moves until one is legal
                    for sim_pos in positions:
                        legal, r = board.moveLegal(f"{sim_pos}{endPos}", whiteMove)
                        if legal:
                            startPos = sim_pos

            elif len(positions) == 1:
                startPos = positions[0]

            if startPos == None:
                # Full game and move details printed for diagnosis
                for turn in output:
                    print(f"{board.render(turn['board'])}\nMove: {turn['move']}")
                raise Exception(
                    f"Game: {game}\nCannot find piece that goes to {endPos} whiteMove: {whiteMove}\n{board.render()}\nGame: {raw_moves}\nlast move: {move}\nPiece Channel: {piece_channel}\nPiece Type Mask: {pieceTypeMask}\nAll Piece Positions: {[str(p) for p in all_piece_positions]}\nPositions: {[str(p) for p in positions]}\ncanSee: {[str(p) for p in canSee]}"
                )

        if promotion == None:
            promotion = ""

        if startPos == None:
            raise Exception("Start piece still not found")

        # Carry out update
        translatedMove = f"{startPos}{endPos}{promotion.lower()}"

        output.append(
            {
                "board": board.getBoardCopy(),
                "params": board.getParams(),
                "move": translatedMove,
                "eval": evals[m_i],
            }
        )

        # Update board after recording previous position, so that each move corresponds to previous board state
        board.update(translatedMove, whiteMove)

        whiteMove = not whiteMove  # Alternate after each move

    return output

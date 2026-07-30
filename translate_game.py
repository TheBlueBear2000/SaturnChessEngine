import re
import saturn_shared

PIECE_CHANNELS = [None, "R", "N", "B", "Q", "K"]


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
        evals.append(float(we))
        evals.append(float(be))

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
        (
            castling,
            piece,
            disambiguation,
            capture,
            destination,
            promotion,
            checkOrMate,
        ) = move_pattern.match(move).groups()

        if castling == "O-O-O":
            if whiteMove:
                startPos = saturn_shared.Position("E", 1)
                endPos = saturn_shared.Position("C", 1)
            else:
                startPos = saturn_shared.Position("E", 8)
                endPos = saturn_shared.Position("C", 8)
        elif castling == "O-O":
            if whiteMove:
                startPos = saturn_shared.Position("E", 1)
                endPos = saturn_shared.Position("G", 1)
            else:
                startPos = saturn_shared.Position("E", 8)
                endPos = saturn_shared.Position("G", 8)

        else:
            endPos = saturn_shared.Position(destination[0], destination[1])
            if whiteMove:
                possiblePieces = board.state[:6]
            else:
                possiblePieces = board.state[6:]
            pieceTypeMask = possiblePieces[PIECE_CHANNELS.index(piece)]
            all_piece_positions = board.getMaskPositions(pieceTypeMask)

            # Filter by disambiguation
            positions = []
            for pos in all_piece_positions:
                if disambiguation in str(pos):
                    positions.append(pos)

            # If there are still multiple candidates, check which can attack the square depending on type
            if len(positions) > 1:
                attackers = board.posAttackedBy(endPos, whiteMove, board)
                for pos in positions:
                    if pos in attackers:
                        startPos = pos

            elif len(positions) == 1:
                startPos = positions[0]
            else:
                raise Exception(f"Cannot find piece that goes to {str(endPos)}")

        if promotion == None:
            promotion = ""

        # Carry out update
        translatedMove = f"{startPos}{endPos}{promotion.lower()}"
        board.update(translatedMove, whiteMove)
        output.append(
            {
                "board": board.getBoardCopy(),
                "params": board.getParams(),
                "move": translatedMove,
                "evaluation": evals[m_i],
            }
        )

        whiteMove = not whiteMove  # Alternate after each move

    return output

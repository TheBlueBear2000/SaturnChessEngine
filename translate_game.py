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

    print(f"Game: {raw_moves}")

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
        print(f"\n\nMOVE: {move}, WHITE: {whiteMove}")
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
            piece_channel = PIECE_CHANNELS.index(piece)
            pieceTypeMask = possiblePieces[piece_channel]
            all_piece_positions = board.getMaskPositions(pieceTypeMask)
            print(f"All piece positions: {[str(p) for p in all_piece_positions]}")

            # Filter by disambiguation
            positions = []
            print(f"Disambiguation: {disambiguation}")
            for pos in all_piece_positions:
                if disambiguation in str(pos).lower():
                    positions.append(pos)
            print(f"Filtered positions: {[str(p) for p in positions]}")

            # If there are still multiple candidates, check which can attack the square depending on type
            if len(positions) > 1:
                canSee = board.posAttackedBy(endPos, whiteMove, board, forTake=capture)
                print(
                    f"canSee: {[str(attacker) for attacker in canSee]}\nPositions: {[str(position) for position in positions]}\nEndPos: {str(endPos)}\nBoard State:"
                )

                print(board.render())
                print(
                    f"canSee type: {type(canSee[0])}, positions type: {type(positions[0])}"
                )
                for pos in positions:
                    if pos in canSee:
                        startPos = pos

                if startPos == None:
                    raise Exception(
                        f"Cannot find any piece to go to {endPos} in channel {piece_channel}"
                    )

            elif len(positions) == 1:
                startPos = positions[0]
            else:
                raise Exception(f"Cannot find piece that goes to {str(endPos)}")

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
                "evaluation": evals[m_i],
            }
        )

        # Update board after recording previous position, so that each move corresponds to previous board state
        board.update(translatedMove, whiteMove)

        whiteMove = not whiteMove  # Alternate after each move

    return output

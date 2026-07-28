class Board:
    def __init__(self):
        self.state = [
            0b0000000000000000000000000000000000000000000000001111111100000000,  # White pawns
            0b0000000000000000000000000000000000000000000000000000000010000001,  # White rooks
            0b0000000000000000000000000000000000000000000000000000000001000010,  # White knights
            0b0000000000000000000000000000000000000000000000000000000000100100,  # White bishops
            0b0000000000000000000000000000000000000000000000000000000000010000,  # White queen
            0b0000000000000000000000000000000000000000000000000000000000001000,  # White king
            0b0000000011111111000000000000000000000000000000000000000000000000,  # Black pawns
            0b1000000100000000000000000000000000000000000000000000000000000000,  # Black rooks
            0b0100001000000000000000000000000000000000000000000000000000000000,  # Black knights
            0b0010010000000000000000000000000000000000000000000000000000000000,  # Black bishops
            0b0001000000000000000000000000000000000000000000000000000000000000,  # Black queen
            0b0000100000000000000000000000000000000000000000000000000000000000,  # Black king
        ]
        self.whiteCastled, self.blackCastled = False, False

    def squareToInt(
        self, square
    ):  # Assumes a form of letter number, such as a3, c7, etc
        return ord(lower(square[0])) + (8 * (int(square[1]) - 1))

    def positionMask(self, square):
        return 1 << self.squareToInt(square)

    def movePiece(self, startPos, endPos, channel):
        startMask = self.positionMask(startPos)
        endMask = self.positionMask(endPos)
        self.state[channel] &= ~startMask  # Remove piece from start pos
        self.state[channel] |= endMask  # Add piece back to end pos

    def get_board(self):
        return self.state

    # def moveLegal(move: str) # Checks move legality

    def update(self, move: str, whiteMove: bool):  # Assumes all moves are legal

        capture = "x" in move  # Boolean flag for capture
        move.replace("x", "-")  # Now default captures to normal moves
        piece = move.pop(0)  # Remove and store piece type
        positions = move.split("-")  # Split into beginning and end pos

        # Castling
        if "O" in positions:
            if len(positions) == 3:  # Long castle
                if whiteMove:
                    self.movePiece("e1", "c1", 5)  # Move king
                    self.movePiece("a1", "d1", 1)  # Move rook
                else:
                    self.movePiece("e8", "c8", 5)  # Move king
                    self.movePiece("a8", "d8", 1)  # Move rook
            else:  # Short castle
                if whiteMove:
                    self.movePiece("e1", "g1", 5)  # Move king
                    self.movePiece("h1", "f1", 1)  # Move rook
                else:
                    self.movePiece("e8", "c8", 5)  # Move king
                    self.movePiece("a8", "d8", 1)  # Move rook
            return

        # Promotion

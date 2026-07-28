CHANNEL_ORDER = ["R", "N", "B", "Q", "K"]
CHANNEL_VALUES = [1, 5, 3, 3, 10]  # Includes pawn at start, excludes king at end


class Board:
    def __init__(self):
        self.state = [
            0b0000000000000000000000000000000000000000000000001111111100000000,  # White pawns
            0b0000000000000000000000000000000000000000000000000000000010000001,  # White rooks
            0b0000000000000000000000000000000000000000000000000000000001000010,  # White knights
            0b0000000000000000000000000000000000000000000000000000000000100100,  # White bishops
            0b0000000000000000000000000000000000000000000000000000000000001000,  # White queen
            0b0000000000000000000000000000000000000000000000000000000000010000,  # White king
            0b0000000011111111000000000000000000000000000000000000000000000000,  # Black pawns
            0b1000000100000000000000000000000000000000000000000000000000000000,  # Black rooks
            0b0100001000000000000000000000000000000000000000000000000000000000,  # Black knights
            0b0010010000000000000000000000000000000000000000000000000000000000,  # Black bishops
            0b0000100000000000000000000000000000000000000000000000000000000000,  # Black queen
            0b0001000000000000000000000000000000000000000000000000000000000000,  # Black king
        ]
        self.whiteCanCastle, self.blackCanCastle = False, False
        self.whiteScore, self.blackScore = 0, 0

    def identifyChannelFromMove(self, move: str, whiteMove: bool):
        if move[1].isnumeric():  # For pawns
            return 6 * int(whiteMove), move
        piece = move.pop(0)
        return (CHANNEL_ORDER.index(piece) + 1) + (6 * int(whiteMove)), move

    def squareToInt(
        self, square
    ):  # Assumes a form of letter number, such as a3, c7, etc
        return ord(square[0].lower()) + (8 * (int(square[1]) - 1))

    def positionMask(self, square):
        return 1 << self.squareToInt(square)

    def movePiece(self, startPos, endPos, channel):
        startMask = self.positionMask(startPos)
        endMask = self.positionMask(endPos)
        self.state[channel] &= ~startMask  # Remove piece from start pos
        self.state[channel] |= endMask  # Add piece back to end pos

    def getPieceValue(self, channel):
        modular_channel = channel % 6
        if modular_channel == 5:
            raise Exception("Cannot get king value")
        return CHANNEL_VALUES[modular_channel]

    def takePiece(self, takePos, whiteMove):
        takeMask = self.positionMask(takePos)
        for c, channel in enumerate(self.state):
            if takeMask & channel:
                self.state[c] &= ~takeMask
                score = self.getPieceValue(c)
                if whiteMove:
                    self.whiteScore += score
                else:
                    self.blackScore += score
                return
        raise Exception(
            f"Move alledges to capture piece, but there is no piece to capture at {endPos}"
        )

    def get_board(self):
        return self.state

    # def moveLegal(move: str) # Checks move legality

    def update(self, move: str, whiteMove: bool):  # Assumes all moves are legal
        # Remove check and checkmate notation
        move = move.strip("#")
        move = move.strip("+")

        # Flag and remove enpassant notation
        enpassant = move.endswith("e.p.")
        move = move.strip("e.p.")
        move = move.strip()  # incase of trailing whitespace after e.p. removal

        # Flag and remove capture notation
        capture = "x" in move
        move.replace("x", "-")

        # Flag and remove promotion notation
        move.split("=")
        promoteTo = ""
        if len(move) > 1:
            promoteTo = move[1]
        move = move[0]

        # Castling
        if move.upper() == "O-O-O":  # Long castle
            if whiteMove:
                self.movePiece("e1", "c1", 5)  # Move king
                self.movePiece("a1", "d1", 1)  # Move rook
            else:
                self.movePiece("e8", "c8", 5)  # Move king
                self.movePiece("a8", "d8", 1)  # Move rook
            return
        elif move.upper() == "O-O":  # Short castle
            if whiteMove:
                self.movePiece("e1", "g1", 5)  # Move king
                self.movePiece("h1", "f1", 1)  # Move rook
            else:
                self.movePiece("e8", "c8", 5)  # Move king
                self.movePiece("a8", "d8", 1)  # Move rook
            return

        # If move is not castling, it follows standard form
        # Split move into beginning and end positions
        positions = move.split("-")  # Split into beginning and end pos
        startPos, endPos = positions[0], positions[1]

        # Validate position form
        if not (
            len(startPos) == 2
            and "a" <= startPos[0] <= "h"
            and "1" <= startPos[1] <= "8"
        ):
            raise Exception(f"Start Pos is invalid: {startPos}")
        if not (
            len(endPos) == 2 and "a" <= endPos[0] <= "h" and "1" <= endPos[1] <= "8"
        ):
            raise Exception(f"End Pos is invalid: {endPos}")

        # After castling, get piece move
        pieceChannel, move = self.identifyChannelFromMove(
            move, whiteMove
        )  # Remove and store piece type

        # Capture
        if capture:
            if enpassant:
                takePos = (
                    endPos[0] + startPos[1]
                )  # Taken piece is always column of new pos but row of old
                self.takePiece(takePos, whiteMove)
            else:
                self.takePiece(endPos, whiteMove)

        # Standard Move
        self.movePiece(startPos, endPos, pieceChannel)

        # Promotion
        if pieceChannel in [0, 6] and promoteTo != "":
            self.state[channel] &= ~self.positionMask(endPos)  # Delete pawn
            self.state[
                (CHANNEL_ORDER.index(piece) + 1)
                + (6 * int(whiteMove))  # Identify channel of new piece
            ] |= self.positionMask(
                endPos
            )  # Add new piece by channel

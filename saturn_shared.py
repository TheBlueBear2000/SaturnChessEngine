from math import log2
from enum import Enum

CHANNEL_ORDER = ["R", "N", "B", "Q", "K"]
CHANNEL_VALUES = [1, 5, 3, 3, 10]  # Includes pawn at start, excludes king at end


class File(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7


class Position:
    def __init__(self, file, rank):
        self.file = File(ord(file.lower()) - 97)
        self.rank = int(rank) - 1

    def isBishopAligned(self, otherPos):
        return (
            self.rank + self.file.value == otherPos.rank + otherPos.file.value
            or self.rank - self.file.value == otherPos.rank - otherPos.file.value
        )

    def isRookAligned(self, otherPos):
        return otherPos.file == self.file or otherPos.rank == self.rank


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
        (
            self.whiteCanLongCastle,
            self.whiteCanShortCastle,
            self.blackCanLongCastle,
            self.blackCanShortCastle,
        ) = (True, True, True, True)

        self.whiteScore, self.blackScore = 0, 0

        self.previousMovePawnPushedFile = None

    def identifyChannelFromPos(self, pos: Position):
        mask = self.positionToMask(pos)
        for c, channel in enumerate(self.state):
            if mask & channel:
                return c

    def positionToInt(self, position: Position):
        # Assumes a form of letter number, such as a3, c7, etc
        return (position.file.value) + (8 * position.rank)

    def intToPosition(self, position: int):
        return Position(chr((position % 8) + 97), position // 8)

    def getMaskPositions(self, mask: int):
        positions = []
        for i in range(64):
            testMask = 1 << i
            if testMask & mask:
                positions.append(self.intToPosition(i))
        return positions

    def positionToMask(self, position: Position):
        return 1 << self.positionToInt(position)

    def movePiece(self, startPos: Position, endPos: Position, channel: int):
        startMask = self.positionToMask(startPos)
        endMask = self.positionToMask(endPos)
        self.state[channel] &= ~startMask  # Remove piece from start pos
        self.state[channel] |= endMask  # Add piece back to end pos

    def getPieceValue(self, channel: int):
        modular_channel = channel % 6
        if modular_channel == 5:
            raise Exception("King has no value")
        return CHANNEL_VALUES[modular_channel]

    def takePiece(self, takePos: Position, whiteMove: bool):
        takeMask = self.positionToMask(takePos)
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

    def stripMove(self, move: str, whiteMove: bool):
        startPos = Position(move[0], move[1])
        endPos = Position(move[2], move[3])

        promoteTo = ""
        if len(move) == 5:
            promoteTo = move[4].upper()

        # Exctract moved piece channel
        pieceChannel = self.identifyChannelFromPos(startPos)

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

        return (
            startPos,
            endPos,
            pieceChannel,
            promoteTo,
        )

    def getBoard(self):
        return self.state, {self.whiteCanLongCastle}

    def getParams(self):
        return {
            "castling_rights": (
                self.whiteCanLongCastle,
                self.whiteCanShortCastle,
                self.blackCanLongCastle,
                self.blackCanShortCastle,
            ),
            "scores": (
                self.whiteScore,
                self.blackScore,
            ),
            "previousMovePawnPushedFile": self.previousMovePawnPushedFile,
        }

    def setParams(self, params):
        (
            self.whiteCanLongCastle,
            self.whiteCanShortCastle,
            self.blackCanLongCastle,
            self.blackCanShortCastle,
        ) = params["castling_rights"]
        self.whiteScore, self.blackScore = params["scores"]
        self.previousMovePawnPushedFile = params["previousMovePawnPushedFile"]

    def getBoardCopy(self):
        return [channel for channel in self.state]

    def update(
        self, move: str, whiteMove: bool, realUpdate: bool = True
    ):  # Assumes all moves are legal
        (
            startPos,
            endPos,
            pieceChannel,
            promoteTo,
        ) = self.stripMove(move, whiteMove)

        if not realUpdate:
            backupBoard = self.getBoardCopy()
            backupParams = self.getParams()

        # Castling - move rooks first
        if pieceChannel == 5:  # White Castle
            if self.whiteCanLongCastle and endPos.file == File.C:
                self.movePiece(Position("A", 1), Position("D", 1), 1)
                self.whiteCanLongCastle, self.whiteCanShortCastle = False
            if self.whiteCanShortCastle and endPos.file == File.G:
                self.movePiece(Position("H", 1), Position("F", 1), 1)
                self.whiteCanLongCastle, self.whiteCanShortCastle = False
        elif pieceChannel == 11:  # Black Castle
            if self.blackCanLongCastle and endPos.file == File.C:
                self.movePiece(Position("A", 8), Position("D", 8), 7)
                self.blackCanLongCastle, self.blackCanShortCastle = False
            if self.blackCanShortCastle and endPos.file == File.G:
                self.movePiece(Position("H", 8), Position("F", 8), 7)
                self.blackCanLongCastle, self.blackCanShortCastle = False

        # Invalidate castles once rooks are moved
        if pieceChannel == 1:
            if startPos == Position("A", 1):
                self.whiteCanLongCastle = False
            elif startPos == Position("H", 1):
                self.whiteCanShortCastle = False
        elif pieceChannel == 7:
            if startPos == Position("A", 8):
                self.blackCanLongCastle = False
            elif startPos == Position("H", 8):
                self.blackCanShortCastle = False

        # Capture
        if self.identifyChannelFromPos(endPos) != None:
            # En Passant derivation check
            if self.previousMovePawnPushedFile == endPos.file and (
                (whiteMove and pieceChannel == 0 and endPos.rank == 6)
                or ((not whiteMove) and pieceChannel == 6 and endPos.rank == 3)
            ):
                takePos = Position(str(endPos.file.name), startPos.rank)
                self.takePiece(takePos, whiteMove)
            else:
                self.takePiece(endPos, whiteMove)

        # Standard Move
        self.movePiece(startPos, endPos, pieceChannel)

        # Promotion
        if pieceChannel in [0, 6] and promoteTo != "":
            self.state[channel] &= ~self.positionToMask(endPos)  # Delete pawn
            self.state[
                (CHANNEL_ORDER.index(piece) + 1)
                + (6 * int(whiteMove))  # Identify channel of new piece
            ] |= self.positionToMask(
                endPos
            )  # Add new piece by channel

        if realUpdate:
            # Expose double pushed pawns for enpassant
            if pieceChannel in [0, 6] and abs(startPos.rank - endPos.rank) == 2:
                self.previousMovePawnPushedFile = endPos.file
            else:
                self.previousMovePawnPushedFile = None
            # No need to return anything since board is updated

        else:  # If update is not real, return new board and params and reset to old one
            newBoard = self.getBoardCopy()
            newParams = self.getParams()
            self.state = backupBoard
            self.setParams(backupParams)
            return newBoard, newParams

    def isCheck(self, pos: Position, isWhiteCheck: bool, board=self.state):
        if isWhiteCheck:
            attackingPieces = board[6:]
            # Use the fact that there is 1 king and so location must be power of two to identify digit position in mask
            kingPos = self.intToPosition(log2(board[5]))
        else:
            attackingPieces = board[:6]
            kingPos = self.intToPosition(log2(board[11]))

        for pawn in self.getMaskPositions(attackingPieces[0]):
            {}

    def moveLegal(move: str, whiteMove: bool):  # Checks move legality
        (
            startPos,
            endPos,
            pieceChannel,
            enpassant,
            capture,
            promoteTo,
            longCastle,
            shortCastle,
        ) = self.stripMove(move, whiteMove)

        # Check castling legality
        if shortCastle:
            {}

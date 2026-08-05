from util import mask_to_bitmap

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
    def __init__(self, file, rank, fromPositionParts=False):
        if fromPositionParts:
            self.file = file
            self.rank = rank
        else:
            self.file = File(ord(file.lower()) - 97)
            self.rank = int(rank) - 1

    def __str__(self):
        return f"{self.file.name}{self.rank + 1}"

    def __eq__(self, other):
        if other == None:
            return False
        return self.file == other.file and self.rank == other.rank

    def isBishopAligned(self, otherPos):
        return (
            self.rank + self.file.value == otherPos.rank + otherPos.file.value
            or self.rank - self.file.value == otherPos.rank - otherPos.file.value
        )

    def isRookAligned(self, otherPos):
        return otherPos.file == self.file or otherPos.rank == self.rank

    def isKnightAligned(self, otherPos):
        return (
            not self.isRookAligned(
                otherPos
            )  # Check that pieces are not linearly aligned
        ) and (
            abs(self.rank - otherPos.rank) + abs(self.file.value - otherPos.file.value)
            == 3  # Check that manhatten distance is 3
        )

    def isKingAligned(self, otherPos):
        return (
            abs(self.rank - otherPos.rank) <= 1
            and abs(self.file.value - otherPos.file.value) <= 1
        )


class Board:
    def __init__(self, custom_layout=None):
        if custom_layout == None:
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
        else:
            self.state = custom_layout  # Allow custom board setups, but ignore custom params (for now)
        (
            self.whiteCanLongCastle,
            self.whiteCanShortCastle,
            self.blackCanLongCastle,
            self.blackCanShortCastle,
        ) = (True, True, True, True)

        self.whiteScore, self.blackScore = 0, 0

        self.previousMovePawnPushedFile = None

        self.whiteMove = True

    def identifyChannelFromPos(self, pos: Position):
        mask = self.positionToMask(pos)
        for c, channel in enumerate(self.state):
            if mask & channel:
                return c

    def positionToInt(self, position: Position):
        # Assumes a form of letter number, such as a3, c7, etc
        return (position.file.value) + (8 * position.rank)

    def intToPosition(self, intPosition: int):
        return Position(chr((intPosition % 8) + 97), (intPosition // 8) + 1)

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

    def posContainsOwnPiece(self, pos, anyPiece=False, flip=False):
        colour_check = (not self.whiteMove) if flip else self.whiteMove
        if anyPiece:
            pieces = self.state
        elif colour_check:
            pieces = self.state[:6]
        else:
            pieces = self.state[6:]
        checkMask = self.positionToMask(pos)
        for channel in pieces:
            if checkMask & channel:
                return True
        return False

    def render(self, board=None):
        if board == None:
            board = self.state
        image = ""
        channelIcons = ["P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k"]
        for row in range(8):
            line = ""
            for col in range(8):
                bit_index = 63 - (row * 8 + col)
                piece = "."
                for layer, mask in enumerate(board):
                    if (mask >> bit_index) & 1:
                        piece = channelIcons[layer]
                        break
                line += piece
            image += f"{8-row} " + line[::-1] + "\n"
        return image + "  abcdefgh"

    def deepRender(self, board=None):
        channelIcons = ["P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k"]
        if board == None:
            board = self.state
        image = ""
        for channel, mask in enumerate(board):
            image += f"Channel: {channelIcons[channel]}\n"
            bitmap = mask_to_bitmap(mask)
            for row in bitmap:
                image += str(row[::-1]) + "\n"
            image += "\n"
        return image

    # Based on checking from the rook's position - make sure to check that final pos is clear of own-colored pieces
    def rookCanMoveToPos(self, pos: Position, otherPos: Position):
        if pos.isRookAligned(otherPos):
            # Check that final pos does not contain own piece
            if self.posContainsOwnPiece(otherPos):
                return False
            # Final position is either free or occupied by opponent
            # Check for piece inbetween
            if pos.file == otherPos.file:  # Vertical check
                for rank in range(
                    min(pos.rank, otherPos.rank) + 1, max(pos.rank, otherPos.rank)
                ):
                    checkMask = self.positionToMask(
                        Position(pos.file, rank, fromPositionParts=True)
                    )
                    for channel in self.state:
                        if checkMask & channel:
                            return False
                return True
            else:
                for file_i in range(
                    min(pos.file.value, otherPos.file.value) + 1,
                    max(pos.file.value, otherPos.file.value),
                ):
                    checkMask = self.positionToMask(
                        Position(File(file_i), pos.rank, fromPositionParts=True)
                    )
                    for channel in self.state:
                        if checkMask & channel:
                            return False
                return True
        return False

    def knightCanMoveToPos(self, pos: Position, otherPos: Position):
        if pos.isKnightAligned(otherPos):
            # Check that final pos does not contain own piece
            if self.posContainsOwnPiece(otherPos):
                return False
            # If squares are knight aligned and the final position is not occupied by own piece, knight can move there
            return True
        return False

    def bishopCanMoveToPos(self, pos: Position, otherPos: Position):
        if pos.isBishopAligned(otherPos):
            # Check that final pos does not contain own piece
            if self.posContainsOwnPiece(otherPos):
                return False
            # Final position is either free or occupied by opponent
            # Check for piece inbetween
            if pos.rank + pos.file.value == otherPos.rank + otherPos.file.value:
                # Aligned negative diagonal
                diagonal_value = pos.rank + pos.file.value
                for rank in range(
                    min(pos.rank, otherPos.rank) + 1, max(pos.rank, otherPos.rank)
                ):
                    checkMask = self.positionToMask(
                        Position(
                            File(diagonal_value - rank), rank, fromPositionParts=True
                        )
                    )
                    for channel in self.state:
                        if checkMask & channel:
                            return False
                return True
            else:
                # Aligned positive diagonal
                diagonal_value = pos.rank - pos.file.value
                for rank in range(
                    min(pos.rank, otherPos.rank) + 1, max(pos.rank, otherPos.rank)
                ):
                    checkMask = self.positionToMask(
                        Position(
                            File(rank - diagonal_value), rank, fromPositionParts=True
                        )
                    )
                    for channel in self.state:
                        if checkMask & channel:
                            return False
                return True
        return False

    # DOES NOT ACCOUNT FOR CHECK
    def kingCanMoveToPos(self, pos: Position, otherPos: Position):
        if pos.isKingAligned(otherPos):
            # Check that final pos does not contain own piece
            if self.posContainsOwnPiece(otherPos):
                return False
            # Like knight, there is no need to check middling squares since king only moves one square at a time
            return True
        return False

    def takePiece(self, takePos: Position):
        takeMask = self.positionToMask(takePos)
        for c, channel in enumerate(self.state):
            if takeMask & channel:
                self.state[c] &= ~takeMask
                score = self.getPieceValue(c)
                if self.whiteMove:
                    self.whiteScore += score
                else:
                    self.blackScore += score
                return
        raise Exception(
            f"Move alledges to capture piece, but there is no piece to capture at {takePos}, whiteMove: {self.whiteMove}\n{self.render()}"
        )

    def stripMove(self, move: str):
        move = move.lower()
        # Validate position form
        if not (
            "a" <= move[0] <= "h"
            and "1" <= move[1] <= "8"
            and "a" <= move[2] <= "h"
            and "1" <= move[3] <= "8"
        ):
            raise Exception(f"Move is invalid: {move}")

        startPos = Position(move[0], move[1])
        endPos = Position(move[2], move[3])

        promoteTo = ""
        if len(move) == 5:
            promoteTo = move[4].upper()

        # Exctract moved piece channel
        pieceChannel = self.identifyChannelFromPos(startPos)

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

    def update(self, move: str, realUpdate: bool = True):  # Assumes all moves are legal
        (
            startPos,
            endPos,
            pieceChannel,
            promoteTo,
        ) = self.stripMove(move)

        if not realUpdate:
            backupBoard = self.getBoardCopy()
            backupParams = self.getParams()

        # Castling - move rooks first
        if pieceChannel == 5:  # White Castle
            if self.whiteCanLongCastle and endPos.file == File.C:
                self.movePiece(Position("A", 1), Position("D", 1), 1)
            if self.whiteCanShortCastle and endPos.file == File.G:
                self.movePiece(Position("H", 1), Position("F", 1), 1)
                # Invalidate castle after king has moved
            self.whiteCanLongCastle, self.whiteCanShortCastle = False, False

        elif pieceChannel == 11:  # Black Castle
            if self.blackCanLongCastle and endPos.file == File.C:
                self.movePiece(Position("A", 8), Position("D", 8), 7)
            if self.blackCanShortCastle and endPos.file == File.G:
                self.movePiece(Position("H", 8), Position("F", 8), 7)
                # Invalidate castle after king has moved
            self.blackCanLongCastle, self.blackCanShortCastle = False, False

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
        # Invalidate castles once rooks are taken
        # If a piece is able to move to a castle's location,
        # the castle is taken or it has already moved
        if endPos == Position("A", 1):
            self.whiteCanLongCastle = False
        elif endPos == Position("H", 1):
            self.whiteCanShortCastle = False
        elif endPos == Position("A", 8):
            self.blackCanLongCastle = False
        elif endPos == Position("H", 8):
            self.blackCanShortCastle = False

        # Capture
        if self.identifyChannelFromPos(endPos) != None:
            self.takePiece(endPos)

        # En Passant derivation check
        if self.previousMovePawnPushedFile == endPos.file and (
            (self.whiteMove and pieceChannel == 0 and endPos.rank == 5)
            or ((not self.whiteMove) and pieceChannel == 6 and endPos.rank == 2)
        ):
            takePos = Position(endPos.file, startPos.rank, fromPositionParts=True)
            self.takePiece(takePos)

        # Standard Move
        self.movePiece(startPos, endPos, pieceChannel)

        # Promotion
        if pieceChannel in [0, 6] and promoteTo != "":
            self.state[pieceChannel] &= ~self.positionToMask(endPos)  # Delete pawn
            self.state[
                (CHANNEL_ORDER.index(promoteTo) + 1)
                + (6 * int(not self.whiteMove))  # Identify channel of new piece
            ] |= self.positionToMask(
                endPos
            )  # Add new piece by channel

        if realUpdate:
            # Expose double pushed pawns for enpassant
            if pieceChannel in [0, 6] and abs(startPos.rank - endPos.rank) == 2:
                self.previousMovePawnPushedFile = endPos.file
            else:
                self.previousMovePawnPushedFile = None
            self.whiteMove = not self.whiteMove  # Alternate move owner
            # No need to return anything since board is updated

        else:  # If update is not real, return new board and params and reset to old one
            newBoard = self.getBoardCopy()
            newParams = self.getParams()
            self.state = backupBoard
            self.setParams(backupParams)
            return newBoard, newParams

    # Note: doesnt check for enpassant threats
    def posAttackedBy(
        self,
        pos: Position,
        board=None,
        quitAtOne: bool = False,
        forTake: bool = True,
        flip: bool = False,
    ):
        if board == None:
            board = self
        elif type(board) != Board:
            board = Board(board)
            board.whiteMove = self.whiteMove

        attackingShortlist = []

        # if checkingOwnPieces == None:
        #     pieceChannel = board.identifyChannelFromPos(pos)
        # else:
        #     isWhite = not (
        #         (checkingOwnPieces and board.whiteMove)
        #         or not (checkingOwnPieces or board.whiteMove)
        #     )
        #     if flip:
        #         isWhite = not isWhite

        #     pieceChannel = 6 * int(
        #         isWhite
        #     )  # Will be 0 if white move (therefore finding white pieces), or 6 if black move (therefore finding black pieces)
        # if pieceChannel < 6:
        #     checkingWhite = True
        #     attackingPieces = board.state[:6]
        #     # Use the fact that there is 1 king and so location must be power of two to identify digit position in mask
        # else:
        #     checkingWhite = False
        #     attackingPieces = board.state[6:]

        checkingWhite = self.whiteMove
        if flip:
            checkingWhite = not checkingWhite
            board.whiteMove = not board.whiteMove

        if checkingWhite:
            attackingPieces = board.state[:6]
        else:
            attackingPieces = board.state[6:]

        # Pawns
        for pawn in board.getMaskPositions(attackingPieces[0]):
            moveDirection = 1 if checkingWhite else -1
            nextRank = pawn.rank + moveDirection
            if (
                forTake
                and nextRank == pos.rank
                and abs(pos.file.value - pawn.file.value) == 1
            ):
                attackingShortlist.append(pawn)
                if quitAtOne:
                    return True
            elif pos.file.value - pawn.file.value == 0 and not (  # If just pushing pawn
                forTake
                or board.posContainsOwnPiece(
                    Position(pos.file, nextRank, True), anyPiece=True
                )
            ):
                # First check for double push
                if pos.rank - pawn.rank == 2 * moveDirection and (
                    not board.posContainsOwnPiece(
                        Position(pos.file, nextRank + moveDirection, True),
                        anyPiece=True,
                    )
                ):
                    attackingShortlist.append(pawn)
                    if quitAtOne:
                        return True
                # If not, check for single push
                elif pos.rank - pawn.rank == moveDirection:
                    attackingShortlist.append(pawn)
                    if quitAtOne:
                        return True

        # Rooks
        for rook in board.getMaskPositions(attackingPieces[1]):
            if board.rookCanMoveToPos(rook, pos):
                attackingShortlist.append(rook)
                if quitAtOne:
                    return True

        # Knights
        for knight in board.getMaskPositions(attackingPieces[2]):
            if board.knightCanMoveToPos(knight, pos):
                attackingShortlist.append(knight)
                if quitAtOne:
                    return True

        # Bishops
        for bishop in board.getMaskPositions(attackingPieces[3]):
            if board.bishopCanMoveToPos(bishop, pos):
                attackingShortlist.append(bishop)
                if quitAtOne:
                    return True

        # Queens
        for queen in board.getMaskPositions(attackingPieces[4]):
            if board.bishopCanMoveToPos(queen, pos) or board.rookCanMoveToPos(
                queen, pos
            ):
                attackingShortlist.append(queen)
                if quitAtOne:
                    return True

        # King
        king = board.getMaskPositions(attackingPieces[5])[0]
        if board.kingCanMoveToPos(king, pos):
            attackingShortlist.append(king)
            if quitAtOne:
                return True

        if quitAtOne:
            return False
        return attackingShortlist

    # Checks move legality, returns boolean + error message
    def moveLegal(self, move: str):
        (
            startPos,
            endPos,
            pieceChannel,
            promoteTo,
        ) = self.stripMove(move)

        # Check that startPos contains own piece
        movingOwnPiece = False
        if self.whiteMove:
            ownPieces = self.state[:6]
        else:
            ownPieces = self.state[6:]
        startMask = self.positionToMask(startPos)
        for channel in ownPieces:
            if channel & startMask:
                movingOwnPiece = True
                break
        if not movingOwnPiece:
            return False, f"Start position {str(startPos)} does not contain own piece"

        # Check that end pos doesnt contain own piece
        if self.posContainsOwnPiece(endPos):
            return (
                False,
                f"Cannot move to {endPos} as it is already occupied by own piece",
            )

        # PAWNS
        if pieceChannel == 0:  # White pawn
            if startPos.rank == 1 and endPos.rank == 3 and startPos.file == endPos.file:
                # Pawn is trying to double push, check that there are no pieces between or on
                if self.posContainsOwnPiece(
                    endPos, anyPiece=True
                ) or self.posContainsOwnPiece(
                    Position(endPos.file, endPos.rank - 1, True), anyPiece=True
                ):
                    return False, f"Double pawn push to {endPos} is blocked"

            elif endPos.rank - startPos.rank == 1:  # If pawn is moving 1 forward
                if (
                    endPos.file == startPos.file
                ):  # If pawn is pushing forward on same file
                    if self.posContainsOwnPiece(endPos, anyPiece=True):
                        return False, f"Pawn push to {endPos} is blocked"

                else:  # Pawn is shifting file, implies a take
                    # Check that file is 1 away
                    if abs(startPos.file.value - endPos.file.value) == 1:
                        # Dissallow if position doesn't contain enemy piece and preceding rank doesnt contain enpassantable piece
                        if not (
                            self.posContainsOwnPiece(
                                endPos, flip=False
                            )  # Check for enemy piece
                            or (
                                self.previousMovePawnPushedFile == endPos.file
                                and endPos.rank == 5
                            )
                        ):
                            return (
                                False,
                                f"Pawn has no piece to take at {endPos} (including by enpassant)",
                            )
            else:
                return False, f"Illegal pawn move from {startPos} to {endPos}"

        elif pieceChannel == 6:  # Black pawn
            if startPos.rank == 6 and endPos.rank == 4 and startPos.file == endPos.file:
                # Pawn is trying to double push, check that there are no pieces between or on
                if self.posContainsOwnPiece(  # Check target log
                    endPos, anyPiece=True
                ) or self.posContainsOwnPiece(  # Check between start and end
                    Position(endPos.file, endPos.rank + 1, True), anyPiece=True
                ):
                    return False, f"Double pawn push to {endPos} is blocked"

            elif startPos.rank - endPos.rank == 1:  # If pawn is moving 1 forward
                if (
                    endPos.file == startPos.file
                ):  # If pawn is pushing forward on same file
                    if self.posContainsOwnPiece(endPos, anyPiece=True):
                        return False, f"Pawn push to {endPos} is blocked"

                else:  # Pawn is shifting file, implies a take
                    # Check that file is 1 away
                    if abs(startPos.file.value - endPos.file.value) == 1:
                        # Dissallow if position doesn't contain enemy piece and preceding rank doesnt contain enpassantable piece
                        if not (
                            self.posContainsOwnPiece(
                                endPos, anyPiece=True
                            )  # Check for enemy piece
                            or (
                                self.previousMovePawnPushedFile == endPos.file
                                and endPos.rank == 2
                            )
                        ):
                            return (
                                False,
                                f"Pawn has no piece to take at {endPos} (including by enpassant)",
                            )
            else:
                return False, f"Illegal pawn move from {startPos} to {endPos}"

        # ROOKS
        elif pieceChannel in [1, 7]:
            if not self.rookCanMoveToPos(startPos, endPos):
                return False, f"Rook cannot move to {endPos}"

        # KNIGHTS
        elif pieceChannel in [2, 8]:
            if not self.knightCanMoveToPos(startPos, endPos):
                return False, f"Knight cannot move to {endPos}"

        # BISHOPS
        elif pieceChannel in [3, 9]:
            if not self.bishopCanMoveToPos(startPos, endPos):
                return False, f"Bishop cannot move to {endPos}"

        # QUEENS
        elif pieceChannel in [4, 10]:
            if not (
                self.bishopCanMoveToPos(startPos, endPos)
                or self.rookCanMoveToPos(startPos, endPos)
            ):
                return False, f"Queen cannot move to {endPos}"

        # KINGS
        elif pieceChannel in [5, 11]:
            if not (
                self.kingCanMoveToPos(startPos, endPos)
                or (  # Account for castling
                    (  # Cannot castle while in check
                        not self.posAttackedBy(startPos, quitAtOne=True)
                    )
                    and (
                        (  # White long castle
                            pieceChannel == 5
                            and self.whiteCanLongCastle
                            and startPos == Position("E", 1)
                            and endPos == Position("C", 1)
                            and (
                                not self.posContainsOwnPiece(
                                    Position("B", 1), anyPiece=True
                                )
                            )
                            and (
                                not self.posContainsOwnPiece(
                                    Position("C", 1), anyPiece=True
                                )
                            )
                            and (
                                not self.posContainsOwnPiece(
                                    Position("D", 1), anyPiece=True
                                )
                            )
                        )
                        or (  # White short castle
                            pieceChannel == 5
                            and self.whiteCanShortCastle
                            and startPos == Position("E", 1)
                            and endPos == Position("G", 1)
                            and (
                                not self.posContainsOwnPiece(
                                    Position("F", 1), anyPiece=True
                                )
                            )
                            and (
                                not self.posContainsOwnPiece(
                                    Position("G", 1), anyPiece=True
                                )
                            )
                        )
                        or (  # Black long castle
                            pieceChannel == 11
                            and self.blackCanLongCastle
                            and startPos == Position("E", 8)
                            and endPos == Position("C", 8)
                            and (
                                not self.posContainsOwnPiece(
                                    Position("B", 8), anyPiece=True
                                )
                            )
                            and (
                                not self.posContainsOwnPiece(
                                    Position("C", 8), anyPiece=True
                                )
                            )
                            and (
                                not self.posContainsOwnPiece(
                                    Position("D", 8), anyPiece=True
                                )
                            )
                        )
                        or (  # Black short castle
                            pieceChannel == 11
                            and self.blackCanShortCastle
                            and startPos == Position("E", 8)
                            and endPos == Position("G", 8)
                            and (
                                not self.posContainsOwnPiece(
                                    Position("F", 8), anyPiece=True
                                )
                            )
                            and (
                                not self.posContainsOwnPiece(
                                    Position("G", 8), anyPiece=True
                                )
                            )
                        )
                    )
                )
            ):
                return False, f"King cannot move to {endPos}"
            # Cannot move into check
            if self.posAttackedBy(endPos, quitAtOne=True):
                return False, f"Moving king to {endPos} will place it in check"

        # Test for revealed checks
        temp_board_state, _ = self.update(move, realUpdate=False)  # Simulate move
        kingPosition = self.getMaskPositions(
            temp_board_state[5 if self.whiteMove else 11]
        )[0]

        attackers = self.posAttackedBy(kingPosition, board=temp_board_state, flip=True)

        if self.posAttackedBy(
            kingPosition, board=temp_board_state, quitAtOne=True, flip=True
        ):
            return (
                False,
                f"Cannot move {startPos} to {endPos} as it leaves king at {kingPosition} in check from attacker(s) at {[str(a) for a in attackers]}. New state is:\n{self.render(temp_board_state)}",
            )

        # Allowing revealed check to happen potentially

        return True, f"Moving {startPos} to {endPos} is legal"

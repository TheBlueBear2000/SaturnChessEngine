from saturn_shared import File, Position
from json import dumps

# Tool script to write all possible moves to a json file, which will
# be used as lookup for the main output head of the neural net

END_FILE = "allowed_moves.json"

output = []
totalPossibleMoves = 0

for s_r in range(8):
    for s_f in File:
        startPos = Position(s_f, s_r, True)
        for e_r in range(8):
            for e_f in File:
                endPos = Position(e_f, e_r, True)
                # If bishops, rooks or knights can reach a position, any piece can
                # (queens can go where bishops/rooks can go, and king and pawn moves
                # are a subset of queen moves)
                if (
                    startPos.isBishopAligned(endPos)
                    or startPos.isRookAligned(endPos)
                    or startPos.isKnightAligned(endPos)
                ) and (
                    str(startPos) != str(endPos)
                ):  # Do not allow moves that do nothing
                    output.append(f"{str(startPos)}{str(endPos)}")
                    totalPossibleMoves += 1

print(f"There are {totalPossibleMoves} possible out of {64 * 64} potential moves")

with open(END_FILE, "w") as file:
    file.write(dumps(output, indent=4))

print("Done!")

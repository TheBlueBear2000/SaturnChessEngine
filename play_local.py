from saturn_engine import Engine

from time import sleep

PLAY_WHITE = False

engine = Engine(silent=True)


def render(engine, move=None):
    moveString = f"\nMove: {move}" if move != None else ""
    # Uncomment for deep board rendering (each bitboard):
    # print(f"{moveString}\n{engine.board.deepRender()}")
    print(f"{moveString}\n{engine.board.render()}")


print("Welcome to Saturn play local!")
render(engine)

# If playing white, make first move, then act asif playing black
if PLAY_WHITE:
    sleep(1)
    move = engine.decideMove()
    engine.playMove(move)
    render(engine, move)

while True:
    # Get player move (in correct format)
    wrongFormatCounter = 0
    while True:
        playerMove = input("\nEnter move: ")
        if not (
            "a" <= playerMove[0] <= "h"
            and "1" <= playerMove[1] <= "8"
            and "a" <= playerMove[2] <= "h"
            and "1" <= playerMove[3] <= "8"
        ):
            print("Incorrect format", end="")
            if wrongFormatCounter > 3:
                print(
                    "Please enter in the format xnxn(p), where x/n is the file/rank of the position, in the order startpos endpos, with an optional 'q', 'n', 'b', 'r' for promotion at the end"
                )
        elif not engine.board.moveLegal(playerMove)[0]:
            print(
                f"Sorry, move {playerMove} is illegal: {engine.board.moveLegal(playerMove)[1]}"
            )
        else:
            break

    # Play move
    engine.playMove(playerMove)

    # Render
    render(engine)

    # Wait for a second
    sleep(1)

    # Return move
    move = engine.decideMove()
    engine.playMove(move)
    render(engine, move)

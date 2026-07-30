from datasets import load_dataset
import translate_game

# Only grab first 100 rows for now
streamed_dataset = load_dataset(
    "Lichess/tournament-chess-games", split="train", streaming=True
)
detailed_dataset = streamed_dataset.take(100)

# Convert the stream subset to a list of games (other information is irrelevant)
dataset = [game["movetext"] for game in list(detailed_dataset)]

print(f"Number of games gathered: {len(dataset)}")

for game in dataset:
    # First each game must be translated into a set of board states for training
    translated_game = translate_game.translate(game)

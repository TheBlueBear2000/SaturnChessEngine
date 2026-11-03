def mask_to_bitmap(value):
    return [
        [(value >> (63 - (row * 8 + col))) & 1 for col in range(8)] for row in range(8)
    ]


import requests


def get_filtered_games(
    where, headers, limit=None, batch_size=100, dataset="Lichess/standard-chess-games"
):

    offset = 0
    retrieved = 0

    while limit is None or retrieved < limit:
        length = batch_size

        if limit is not None:
            length = min(batch_size, limit - retrieved)

        params = {
            "dataset": dataset,
            "config": "default",
            "split": "train",
            "where": where,
            "offset": offset,
            "length": length,
        }

        response = requests.get(
            "https://datasets-server.huggingface.co/filter",
            headers=headers,
            params=params,
        )

        if not response.ok:
            print("Hugging Face API error:")
            print(response.status_code)
            print(response.text)
            response.raise_for_status()

        data = response.json()

        rows = data["rows"]

        if not rows:
            break

        for row in rows:
            yield row["row"]

        retrieved += len(rows)
        offset += len(rows)

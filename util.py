def mask_to_bitmap(value):
    return [
        [(value >> (63 - (row * 8 + col))) & 1 for col in range(8)] for row in range(8)
    ]

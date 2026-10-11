import torch
import torch.nn as nn
import torch.nn.functional as F

BITBOARD_CHANNELS = 12


class SaturnNetwork(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        # Image encoder
        self.conv = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.image_fc = nn.Linear(32 * 2 * 2, 64)

        # Extra parameters
        self.param_fc = nn.Sequential(nn.Linear(6, 32), nn.ReLU())

        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        # Embedding
        self.move_embedding = nn.Embedding(num_embeddings=n_outputs, embedding_dim=64)

        # Heads
        self.category_head = nn.Linear(64, 4)

    def forward(self, image, params):

        x = self.conv(image)
        x = torch.flatten(x, 1)
        x = F.relu(self.image_fc(x))

        p = self.param_fc(params)

        x = torch.cat([x, p], dim=1)

        x = self.shared(x)

        # Embedding evaluation
        move_embeddings = self.move_embedding.weight
        label_logits = x @ move_embeddings.T

        return (label_logits, self.category_head(x))

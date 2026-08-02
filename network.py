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

        # Heads
        self.label_head = nn.Linear(64, n_outputs)
        self.category_head = nn.Linear(64, 4)
        self.value_head = nn.Linear(64, 1)

    def forward(self, image, params):

        x = self.conv(image)
        x = torch.flatten(x, 1)
        x = F.relu(self.image_fc(x))

        p = self.param_fc(params)

        x = torch.cat([x, p], dim=1)

        x = self.shared(x)

        return (self.label_head(x), self.category_head(x), self.value_head(x))

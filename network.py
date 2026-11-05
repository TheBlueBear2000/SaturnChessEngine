import math
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual

        return self.relu(out)


class SaturnNetwork(nn.Module):

    def __init__(self, n_outputs, n_residual_blocks=4):
        super().__init__()

        # Image encoder
        self.stem = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(n_residual_blocks)]
        )

        # Board projection
        self.board_projection = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1), nn.ReLU(inplace=True)
        )

        # Image representation
        self.image_fc = nn.Sequential(nn.Linear(32 * 8 * 8, 256), nn.ReLU(inplace=True))

        # Extra parameters
        self.param_fc = nn.Sequential(nn.Linear(6, 64), nn.ReLU(inplace=True))

        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # Move embedding
        self.move_embedding = nn.Embedding(num_embeddings=n_outputs, embedding_dim=256)

        # Prediction heads
        self.category_head = nn.Linear(256, 4)

    def forward(self, image, params):
        # Process chessboard
        x = self.stem(image)
        x = self.residual_blocks(x)
        x = self.board_projection(x)

        x = torch.flatten(x, 1)
        x = self.image_fc(x)

        # Process extra parameters
        p = self.param_fc(params)

        # Combine representations
        x = torch.cat([x, p], dim=1)

        # Shared feature extraction
        x = self.shared(x)

        # Move prediction
        move_embeddings = self.move_embedding.weight

        label_logits = (x @ move_embeddings.T) / math.sqrt(256)
        category_logits = self.category_head(x)

        return label_logits, category_logits

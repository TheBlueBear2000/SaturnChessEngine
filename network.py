import math
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x + residual
        return self.relu(x)


class SaturnNetwork(nn.Module):
    def __init__(
        self,
        n_outputs,
        n_params=13,  # 4 castling + 1 turn + 8 en-passant file one-hot
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Project CNN features into transformer embedding dimensions
        self.board_projection = nn.Sequential(
            nn.Conv2d(64, d_model, kernel_size=1), nn.ReLU(inplace=True)
        )

        # Positional embeddings for the 64 chessboard squares
        self.position_embedding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        # Embed the additional game parameters (castling, turn, en-passant file)
        self.param_fc = nn.Sequential(
            nn.Linear(n_params, d_model), nn.ReLU(inplace=True)
        )

        # CLS token used to aggregate information from the board
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.transformer_norm = nn.LayerNorm(d_model)

        # Move prediction head
        self.move_embedding = nn.Embedding(
            num_embeddings=n_outputs, embedding_dim=d_model
        )

        # Promotion prediction head (q, n, b, r)
        self.category_head = nn.Linear(d_model, 4)

    def forward(self, image, params):

        batch_size = image.shape[0]

        # 1. Extract spatial features using the CNN
        x = self.cnn(image)

        # 2. Project CNN features into transformer dimensions
        x = self.board_projection(x)

        # 3. Convert the board into 64 square tokens
        x = x.flatten(2).transpose(1, 2)

        # 4. Add positional information
        x = x + self.position_embedding

        # 5. Embed additional game parameters
        p = self.param_fc(params)

        # 6. Create CLS token and incorporate game parameters
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls = cls + p.unsqueeze(1)

        # 7. Prepend CLS token to the board tokens
        x = torch.cat([cls, x], dim=1)

        # 8. Process tokens through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        x = self.transformer_norm(x)

        # 9. Extract the CLS token
        position = x[:, 0, :]

        # 10. Predict move logits
        move_embeddings = self.move_embedding.weight

        move_logits = (position @ move_embeddings.T) / math.sqrt(
            move_embeddings.shape[-1]
        )

        # 11. Predict promotion logits
        category_logits = self.category_head(position)

        return move_logits, category_logits

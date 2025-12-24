"""
CNN + LSTM model for dynamic hand gesture recognition (sequences of frames).
"""

import torch
from torch import nn


class CNNBackbone(nn.Module):
    """CNN feature extractor used inside CNN+LSTM."""

    def __init__(self, input_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNNLSTM(nn.Module):
    """
    CNN+LSTM model:
    - CNN backbone for spatial feature extraction per frame
    - LSTM for temporal modeling over sequences of features
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = CNNBackbone(input_channels=input_channels, feature_dim=feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.cnn(x)  # (B*T, F)
        feats = feats.view(b, t, -1)  # (B, T, F)
        lstm_out, _ = self.lstm(feats)  # (B, T, H)
        last_step = lstm_out[:, -1, :]
        logits = self.classifier(last_step)
        return logits



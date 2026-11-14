"""
1D Convolutional Neural Network (1D-CNN) for Multi-Class CAN Bus Attack Detection.

Classifies in-vehicle CAN frames into four categories:
  0: Normal
  1: DoS
  2: Fuzzy
  3: Spoofing (RPM / Gear)
"""

import torch
import torch.nn as nn


class CAN1DCNN(nn.Module):
    """
    Fast-converging 1D-CNN architecture designed for 10-feature CAN bus frame
    representations: [CAN_ID, DATA_0..DATA_7, delta_t].
    """

    def __init__(self, input_features: int = 10, num_classes: int = 4):
        super().__init__()
        self.input_features = input_features

        self.conv_block = nn.Sequential(
            # Input shape: (batch_size, 1, input_features)
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # Sequence length: 10 -> 5
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        flattened_size = 128 * (input_features // 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has channel dimension: (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.conv_block(x)
        logits = self.classifier(features)
        return logits


def get_device() -> torch.device:
    """Automatically detect and return CUDA GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

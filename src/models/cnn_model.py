import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3,
                 dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1:  (3, 128, 128) → (32, 64, 64)
            ConvBlock(in_channels, 32),
            # Block 2:  (32, 64, 64)  → (64, 32, 32)
            ConvBlock(32, 64),
            # Block 3:  (64, 32, 32)  → (128, 16, 16)
            ConvBlock(64, 128),
            # Block 4:  (128, 16, 16) → (256, 8, 8)
            ConvBlock(128, 256),
            # Block 5:  (256, 8, 8)   → (512, 4, 4)
            ConvBlock(256, 512),
            # Block 6:  (512, 4, 4)   → (512, 2, 2)
            ConvBlock(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(256, num_classes),
        )

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SignLanguageCNN(num_classes=36, dropout=0.4)
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    print(f"Output shape: {out.shape}")          # (2, 36)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

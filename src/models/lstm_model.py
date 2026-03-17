import torch
import torch.nn as nn


class SignLanguageLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Temporal attention over LSTM outputs
        attn_in = hidden_size * self.directions
        self.attention = nn.Sequential(
            nn.Linear(attn_in, attn_in // 2),
            nn.Tanh(),
            nn.Linear(attn_in // 2, 1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(attn_in),
            nn.Linear(attn_in, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # (B, T, H)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, H*dirs)

        # Attention pooling
        attn_weights = self.attention(lstm_out)      # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, H*dirs)

        logits = self.classifier(context)
        return logits


if __name__ == "__main__":
    model = SignLanguageLSTM(input_size=126, hidden_size=256,
                             num_layers=2, num_classes=20)
    dummy = torch.randn(4, 30, 126)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (4, 20)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

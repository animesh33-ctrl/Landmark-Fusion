import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)                 # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, D)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    def __init__(
        self,
        input_size: int = 126,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 10,
        dropout: float = 0.3,
        max_len: int = 500,
    ):
        super().__init__()

        # Input projection to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Project input
        x = self.input_proj(x)                         # (B, T, D)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, T+1, D)

        # Positional encoding
        x = self.pos_enc(x)                            # (B, T+1, D)

        # Transformer encoder
        x = self.encoder(x)                            # (B, T+1, D)

        # Use [CLS] token output for classification
        cls_output = x[:, 0, :]                        # (B, D)

        logits = self.classifier(cls_output)            # (B, C)
        return logits


if __name__ == "__main__":
    model = SignLanguageTransformer(
        input_size=126, d_model=128, nhead=8,
        num_layers=4, dim_feedforward=512, num_classes=20,
    )
    dummy = torch.randn(4, 30, 126)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (4, 20)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

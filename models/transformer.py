import torch
import torch.nn as nn
class IntrusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_ff=128,
        dropout=0.25
    ):
        super().__init__()
        self.feature_embed = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, input_dim + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
       
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # x: (B, input_dim)
        x = x.unsqueeze(-1)             
        x = self.feature_embed(x)
        batch_size = x.size(0)
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        # Transformer encoding
        x = self.encoder(x)
        # Take CLS token
        cls_output = x[:, 0]
        # Gated refinement
        cls_output = self.gate(cls_output)
        # Classification
        logits = self.classifier(cls_output)
        return logits
import torch
import torch.nn as nn

# 당신이 이미 만든 DGL 레이어 파일에서 import
# (파일명이 dgl_layers_full.py라고 가정)
from dgl_layers_full import DGLFeatureGAT, DGLTemporalGAT


class HyperscanMTADGAT_MultiLabel(nn.Module):
    """
    Input : x [B, W, F=57]
    Output: logits [B, 3]  -> [i, j, k]
    """
    def __init__(
        self,
        n_features: int = 57,
        window_size: int = 150,
        gat_heads: int = 2,
        dropout: float = 0.25,
        alpha: float = 0.2,
        use_gatv2: bool = True,
        gru_hid_dim: int = 150,
        gru_n_layers: int = 1,
        fuse_mode: str = "concat",  # "concat" or "sum"
    ):
        super().__init__()
        self.F = n_features
        self.W = window_size
        self.fuse_mode = fuse_mode

        # Feature GAT: (B,W,F) -> (B,W,F)
        self.feat_gat = DGLFeatureGAT(
            n_features=n_features,
            window_size=window_size,
            out_window_size=window_size,
            num_heads=gat_heads,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
        )

        # Temporal GAT: (B,W,F) -> (B,W,F)
        self.time_gat = DGLTemporalGAT(
            n_features=n_features,
            window_size=window_size,
            out_features=n_features,
            num_heads=gat_heads,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
            band_k=10,
        )

        if fuse_mode == "concat":
            self.fuse_proj = nn.Sequential(
                nn.Linear(2 * n_features, n_features),
                nn.Dropout(dropout),
            )
        elif fuse_mode == "sum":
            self.fuse_proj = nn.Identity()
        else:
            raise ValueError("fuse_mode must be 'concat' or 'sum'")

        # Sequence encoder
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_hid_dim,
            num_layers=gru_n_layers,
            batch_first=True,
            dropout=0.0 if gru_n_layers == 1 else dropout,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hid_dim, 3),  # [i, j, k]
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B,W,F]
        return logits: [B,3]
        """
        h_feat = self.feat_gat(x)   # [B,W,F]
        h_time = self.time_gat(x)   # [B,W,F]

        if self.fuse_mode == "concat":
            h = torch.cat([h_feat, h_time], dim=-1)   # [B,W,2F]
            h = self.fuse_proj(h)                     # [B,W,F]
        else:
            h = h_feat + h_time                       # [B,W,F]

        out, h_last = self.gru(h)                     # h_last: [L,B,H]
        z = h_last[-1]                                # [B,H]
        logits = self.head(z)                         # [B,3]
        return logits

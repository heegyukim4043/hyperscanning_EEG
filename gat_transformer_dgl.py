import torch
import torch.nn as nn

from modules import ConvLayer, GRULayer, Forecasting_Model
from module_transformer import TransformerReconstruction
from dgl_gat_layers import DGLFeatureGAT, DGLTemporalGAT


class MTAD_GAT_DGL_TransformerRecon(nn.Module):
    """
    Conv(optional) + DGL FeatureGAT + DGL TemporalGAT + GRU + (Forecast head) + Transformer Recon
    """
    def __init__(
        self,
        n_features: int,
        window_size: int,
        out_dim: int,
        kernel_size: int = 7,
        gru_hid_dim: int = 150,
        forecast_hid_dim: int = 150,
        dropout: float = 0.25,

        # DGL GAT params
        gat_heads: int = 4,
        gat_head_dim: int = 8,
        gat_feat_drop: float = 0.1,
        gat_attn_drop: float = 0.1,

        # Transformer recon params
        recon_d_model: int = 128,
        recon_nhead: int = 4,
        recon_num_layers: int = 2,
        recon_dim_ff: int = 256,
        recon_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.out_dim = out_dim

        self.conv = ConvLayer(n_features, kernel_size=kernel_size)

        self.feature_gat = DGLFeatureGAT(
            n_features=n_features, window_size=window_size,
            num_heads=gat_heads, head_dim=gat_head_dim,
            feat_drop=gat_feat_drop, attn_drop=gat_attn_drop
        )
        self.temporal_gat = DGLTemporalGAT(
            n_features=n_features, window_size=window_size,
            num_heads=gat_heads, head_dim=gat_head_dim,
            feat_drop=gat_feat_drop, attn_drop=gat_attn_drop
        )

        self.gru = GRULayer(3 * n_features, gru_hid_dim, 1, dropout)

        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, 1, dropout
        )

        self.recon_model = TransformerReconstruction(
            window_size=window_size,
            enc_dim=gru_hid_dim,
            d_model=recon_d_model,
            nhead=recon_nhead,
            num_layers=recon_num_layers,
            dim_feedforward=recon_dim_ff,
            out_dim=out_dim,
            dropout=recon_dropout,
        )

    def _select_target(self, x: torch.Tensor, target_dims):
        """
        x: [B,W,F]
        return: [B,W,out_dim]
        """
        if target_dims is None:
            return x
        if isinstance(target_dims, int):
            return x[:, :, [target_dims]]
        # list/tuple/np array
        return x[:, :, target_dims]

    def forward(self, x: torch.Tensor, target_dims=None):
        """
        x: [B,W,F]
        """
        # conv (원 repo conv가 [B,W,F]를 받는다고 가정)
        x_conv = self.conv(x) if self.conv is not None else x

        h_feat = self.feature_gat(x_conv)
        h_temp = self.temporal_gat(x_conv)

        h_cat = torch.cat([x_conv, h_feat, h_temp], dim=2)  # [B,W,3F]

        memory, _ = self.gru(h_cat)                         # GRU output sequence expected [B,W,H] (repo 구현 확인 필요)
        # 만약 GRULayer가 (out, h_end)에서 out이 [B,W,H]가 아니라면, 여기서 수정 필요.

        h_last = memory[:, -1, :]                           # [B,H]
        preds = self.forecasting_model(h_last)              # [B,out_dim]

        x_tgt = self._select_target(x, target_dims)         # [B,W,out_dim]
        recons = self.recon_model(memory, x_tgt)            # [B,W,out_dim]

        return preds, recons

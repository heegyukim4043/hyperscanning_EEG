# mtad_gat_dgl_full.py
import torch
import torch.nn as nn

from modules import ConvLayer, GRULayer, Forecasting_Model
from module_transformer import TransformerReconstruction  # 업로드 파일명 기준
from dgl_layers_full import DGLFeatureGAT, DGLTemporalGAT


class MTAD_GAT_DGL_Full(nn.Module):
    """
    Full DGL Graph 모델:
    - Feature GAT (DGL)
    - Temporal GAT (DGL)
    - GRU encoder
    - Forecast head (FC)
    - Reconstruction head (Transformer decoder)
    """
    def __init__(
        self,
        n_features: int,
        window_size: int,
        out_dim: int,
        kernel_size: int = 7,
        use_gatv2: bool = True,
        gat_heads_feat: int = 2,
        gat_heads_time: int = 2,
        gru_n_layers: int = 1,
        gru_hid_dim: int = 150,
        fc_n_layers: int = 1,
        fc_hid_dim: int = 150,
        recon_d_model: int = 128,
        recon_nhead: int = 4,
        recon_num_layers: int = 2,
        recon_dim_ff: int = 256,
        dropout: float = 0.25,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.out_dim = out_dim

        # (선택) Conv: 원본 repo가 conv 결과를 실제로 쓰는지 확인 필요
        # 여기서는 기존 코드 스타일대로 conv 모듈만 두고 사용은 생략 가능
        self.conv = ConvLayer(n_features, kernel_size)

        # DGL GATs
        self.feature_gat = DGLFeatureGAT(
            n_features=n_features,
            window_size=window_size,
            out_window_size=window_size,
            num_heads=gat_heads_feat,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
        )
        self.temporal_gat = DGLTemporalGAT(
            n_features=n_features,
            window_size=window_size,
            out_features=n_features,
            num_heads=gat_heads_time,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
        )

        # GRU encoder: input dim = 3F
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)

        # Forecast head: last hidden -> out_dim
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, fc_hid_dim, out_dim, fc_n_layers, dropout
        )

        # Transformer reconstruction head: memory=[B,W,H], target=[B,W,out_dim]
        self.recon_model = TransformerReconstruction(
            window_size=window_size,
            enc_dim=gru_hid_dim,
            d_model=recon_d_model,
            nhead=recon_nhead,
            num_layers=recon_num_layers,
            dim_feedforward=recon_dim_ff,
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B,W,F]
        returns:
          preds:  [B,out_dim]   (repo의 Forecasting_Model 정의에 따라 [B,out_dim] 또는 [B,1,out_dim]일 수 있음)
          recons: [B,W,out_dim]
        """
        B, W, F = x.shape
        assert W == self.window_size and F == self.n_features, f"x shape mismatch {x.shape}"

        # DGL graphs
        h_feat = self.feature_gat(x)     # [B,W,F]
        h_temp = self.temporal_gat(x)    # [B,W,F]

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # [B,W,3F]

        # GRU encoder outputs sequence
        memory, _ = self.gru(h_cat)      # memory: [B,W,H]  (GRULayer 구현이 batch_first일 때)

        # 일부 repo 구현에서 GRULayer가 (out, h_end)로 반환하며 out이 [W,B,H]일 수도 있음
        # 그 경우 아래 두 줄을 사용:
        if memory.dim() == 3 and memory.shape[0] == W and memory.shape[1] == B:
            memory = memory.transpose(0, 1).contiguous()  # [B,W,H]

        h_last = memory[:, -1, :]        # [B,H]
        preds = self.forecasting_model(h_last)

        # reconstruction target은 out_dim에 맞춰 slice
        x_target = x[:, :, :self.out_dim]  # [B,W,out_dim]
        recons = self.recon_model(memory, x_target)

        return preds, recons

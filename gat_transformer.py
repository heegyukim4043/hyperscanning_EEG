import torch
import torch.nn as nn

from module_transformer_pre import TransformerReconstruction

from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
)

class MTAD_GAT_TransformerRecon(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        out_dim: int,                 # target_dims와 일치하는 출력 차원
        *,
        # --- 인코더/헤드 하이퍼파라미터 ---
        gru_hid_dim: int = 150,
        forecast_hid_dim: int = 150,
        # --- Transformer 재구성 헤드 ---
        recon_d_model: int = 128,
        recon_nhead: int = 4,
        recon_num_layers: int = 2,
        recon_dim_ff: int = 256,
        # --- 공통 ---
        dropout: float = 0.2,
        alpha: float = 0.2,
        feat_gat_embed_dim: int = None,
        time_gat_embed_dim: int = None,
        use_gatv2: bool = True,
        # target dims 전달 (None=전체 복원, int or list 사용 가능)
        target_dims=None,
        kernel_size: int = 7,
        **kwargs
    ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.out_dim = out_dim

        # target_dims 보관 (reconstruction용 teacher forcing에서 사용)
        if target_dims is None:
            self._target_idx = slice(None)            # 전체
        elif isinstance(target_dims, int):
            self._target_idx = [target_dims]
        else:
            self._target_idx = list(target_dims)

        # ----- Frontend -----
        self.conv = ConvLayer(n_features, kernel_size=kernel_size)

        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha,
            feat_gat_embed_dim, use_gatv2
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha,
            time_gat_embed_dim, use_gatv2
        )

        # ----- Sequence Encoder (GRU) -----
        self.gru = GRULayer(3 * n_features, gru_hid_dim, 1, dropout)

        # ----- Forecasting Head (그대로 유지) -----
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, 1, dropout
        )

        # ----- Transformer Reconstruction Head -----
        self.recon_model = TransformerReconstruction(
            window_size=window_size,
            enc_dim=gru_hid_dim,
            d_model=recon_d_model,
            nhead=recon_nhead,
            num_layers=recon_num_layers,
            dim_feedforward=recon_dim_ff,
            out_dim=out_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, W, F]
        returns:
            preds:  [B, out_dim]      (forecast head 출력)
            recons: [B, W, out_dim]   (transformer reconstruction)
        """
        # ----- Frontend: GAT -----
        # (원한다면 conv를 앞에 적용: x = self.conv(x))
        h_feat = self.feature_gat(x)         # [B, W, F]
        h_temp = self.temporal_gat(x)        # [B, W, F]
        h_cat  = torch.cat([x, h_feat, h_temp], dim=2)  # [B, W, 3F]

        # ----- GRU Encoder -----
        memory, _ = self.gru(h_cat)          # 보장: [B, W, H]가 되도록 GRULayer 구현

        # 혹시 GRULayer가 [B, H]만 반환한다면 안전가드
        if memory.dim() == 2:                 # [B, H]
            h_last = memory                   # forecasting용
            memory = h_last.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, W, H]
        else:
            h_last = memory[:, -1, :]         # [B, H]

        # ----- Forecasting -----
        preds = self.forecasting_model(h_last)  # [B, out_dim]

        # ----- Reconstruction (teacher forcing) -----
        # target_dims에 맞게 x를 슬라이스하여 x_target 생성 (shape [B, W, out_dim])
        if isinstance(self._target_idx, slice):
            x_target = x[:, :, self._target_idx]             # 전체
        else:
            x_target = x[:, :, self._target_idx]             # 선택 채널들

        # out_dim과 마지막 차원 일치 보장 (안전가드)
        if x_target.shape[-1] != self.out_dim:
            # 필요 시 projection 대신 단순 절단/확장
            if x_target.shape[-1] > self.out_dim:
                x_target = x_target[:, :, :self.out_dim]
            else:
                # 부족하면 zero-pad (드물지만 방어코드)
                pad = self.out_dim - x_target.shape[-1]
                x_target = torch.cat(
                    [x_target, x_target.new_zeros(x_target.size(0), x_target.size(1), pad)],
                    dim=-1
                )

        recons = self.recon_model(memory, x_target)          # [B, W, out_dim]
        return preds, recons

import torch
import torch.nn as nn
import pandas as pd
import pytorch_forecasting as pyt
#from Autocorr_func import *

from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    RNNDecoder,
    LSTMLayer,
    Forecasting_Model,
    ReconstructionModel,
    Attention
)
from typing import Optional

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)               # [T, D]
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

def _causal_attn_mask(T: int, device):
    # True=mask (upper-tri), False=keep
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

class MTAD_GAT(nn.Module):
    """
    기존 구조에 Transformer 인코더 옵션을 추가한 버전.
    - use_transformer=True이면 GRU 대신 Transformer 인코더 사용
    """
    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2,
        attention_dim=16,
        # ---- Transformer 옵션 ----
        use_transformer: bool = True,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        attn_dropout: float = 0.1,
        causal: bool = True,           # forecasting 용도면 True 권장
        use_last_step: bool = True,    # False면 mean-pool

    ):
        super(MTAD_GAT, self).__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.use_transformer = use_transformer
        self.use_last_step = use_last_step
        self.causal = causal

        # 기존 모듈
        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.attention = Attention(3 * n_features, attention_dim)

        # === 시간 인코더 경로 선택 ===
        if not self.use_transformer:
            # 기존: GRU + (옵션) RNNDecoder
            self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
            self.rnn = RNNDecoder(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
            hid_for_heads = gru_hid_dim
        else:
            # Transformer 인코더 경로
            self.in_proj = nn.Linear(3 * n_features, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=attn_dropout,
                batch_first=True,  # [B, T, D]
                activation="relu",
                norm_first=True,   # Pre-LN
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
            self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max(window_size, 8192))
            hid_for_heads = d_model

        # Heads (기존 그대로 사용)
        self.forecasting_model = Forecasting_Model(hid_for_heads, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, hid_for_heads, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # 기대 입력: x (B, n, k) = (batch, window_size, n_features)
        # 만약 아래 줄 유지가 필요하다면 그대로 두세요 (자체 상관 변환)
        # pytorch_forecasting의 autocorrelation은 축 설정에 주의 필요.
        # x = pyt.utils.autocorrelation(x, dim=0) ; x = torch.sqrt(x)

        # 1) (선택) conv 등을 거치려면 self.conv(x) 사용(현재 주석 처리되어 있었음)
        # x = self.conv(x)

        # 2) GAT들
        h_feat = self.feature_gat(x)   # [B, n, k]
        h_temp = self.temporal_gat(x)  # [B, n, k]

        # 3) concat
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # [B, n, 3k]

        # 4) attention으로 context 벡터와 점수
        #   - 기존 코드는 context만 GRU에 1-step 시퀀스로 넣었는데,
        #     Transformer 경로에서는 보통 전체 시퀀스를 넣고 pooling합니다.
        context, attn_scores = self.attention(h_cat)   # context: [B, 3k]

        if not self.use_transformer:
            # ===== 기존 GRU 경로 =====
            # GRU는 시퀀스를 기대하므로 context를 1-step 시퀀스로 넣는 기존 로직 유지
            _, h_end = self.gru(context.unsqueeze(1))  # h_end: [B, H]
            h_end = h_end.view(x.shape[0], -1)
        else:
            # ===== Transformer 경로 =====
            # (a) proj + posenc
            z = self.in_proj(h_cat)               # [B, n, D]
            z = self.posenc(z)                    # [B, n, D]
            # (b) causal mask (True=mask)
            attn_mask = _causal_attn_mask(z.size(1), z.device) if self.causal else None
            # (c) 인코딩
            z = self.encoder(z, mask=attn_mask)   # [B, n, D]
            # (d) pooling으로 h_end 생성
            if self.use_last_step:
                h_end = z[:, -1, :]               # [B, D]
            else:
                h_end = z.mean(dim=1)             # [B, D]

        # 5) heads
        predictions = self.forecasting_model(h_end)  # [B, n, out_dim] 또는 정의에 맞게
        recons = self.recon_model(h_end)             # [B, n, out_dim]

        return predictions, recons

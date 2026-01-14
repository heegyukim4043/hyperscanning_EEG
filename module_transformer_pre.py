import torch
import torch.nn as nn
import math

# --------------------------
# Positional Encoding
# --------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, W, H]
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# --------------------------
# Transformer Reconstruction Head
# --------------------------
class TransformerReconstruction(nn.Module):
    """
    Transformer 기반 Reconstruction Head
    """
    def __init__(
        self,
        window_size,
        enc_dim,       # GRU hidden dim
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        out_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.out_dim = out_dim

        # Projection
        self.proj_enc = nn.Linear(enc_dim, d_model)
        self.proj_dec_in = nn.Linear(out_dim, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=window_size)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout  # ⚠ batch_first 없음 (PyTorch<1.9)
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

        # 시작 토큰
        self.start_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, memory, x_target):
        """
        memory: [B, W, enc_dim] (GRU output sequence)
        x_target: [B, W, out_dim] (teacher forcing targets)
        """
        B, W, _ = x_target.shape
        device = x_target.device

        # ---- Encoder memory ----
        memory = self.proj_enc(memory)          # [B, W, d_model]
        memory = self.posenc(memory)            # [B, W, d_model]
        memory = memory.transpose(0, 1)         # [W, B, d_model]

        # ---- Decoder input (shifted right) ----
        start = self.start_token.expand(B, 1, self.out_dim)
        tgt_in = torch.cat([start, x_target[:, :-1, :]], dim=1)   # [B, W, out_dim]
        tgt_in = self.proj_dec_in(tgt_in)        # [B, W, d_model]
        tgt_in = self.posenc(tgt_in)             # [B, W, d_model]
        tgt_in = tgt_in.transpose(0, 1)          # [W, B, d_model]

        # ---- Mask ----
        tgt_mask = torch.triu(
            torch.ones(W, W, device=device), diagonal=1
        ).bool()

        # ---- Decoder ----
        dec_out = self.decoder(tgt_in, memory, tgt_mask=tgt_mask)   # [W, B, d_model]
        dec_out = dec_out.transpose(0, 1)                           # [B, W, d_model]
        recons = self.out_proj(dec_out)                             # [B, W, out_dim]
        return recons


# --------------------------
# MTAD-GAT with Transformer Reconstruction
# --------------------------
from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
)

class MTAD_GAT_TransformerRecon(nn.Module):
    def __init__(self, n_features, window_size, out_dim,
                 gru_hid_dim=150,
                 forecast_hid_dim=150,
                 recon_d_model=128,
                 recon_nhead=4,
                 recon_num_layers=2,
                 recon_dim_ff=256,
                 dropout=0.2,
                 alpha=0.2,
                 feat_gat_embed_dim=None,
                 time_gat_embed_dim=None,
                 use_gatv2=True,
                 **kwargs):
        super().__init__()

        # Conv layer
        self.conv = ConvLayer(n_features, kernel_size=7)

        # GAT layers
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha,
            feat_gat_embed_dim, use_gatv2
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha,
            time_gat_embed_dim, use_gatv2
        )

        # GRU encoder
        self.gru = GRULayer(3 * n_features, gru_hid_dim, 1, dropout)

        # Forecasting head
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, 1, dropout
        )

        # Transformer reconstruction head
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

    def forward(self, x):

        target_dims = None
        out_dim =  57 # 57
        x_target = x

        # GAT 전처리
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # [B,T,3F]

        # GRU 인코더
        memory, _ = self.gru(h_cat)  # 가능하면 [B,T,H] 반환하도록 GRULayer 수정

        if memory.dim() == 2:  # [B,H]만 온 경우 대비
            h_last = memory
            memory = h_last.unsqueeze(1).expand(-1, x.size(1), -1)  # [B,T,H]
        else:
            h_last = memory[:, -1, :]

        # Forecast head
        h_last = memory[:, -1, :]
        preds = self.forecasting_model(h_last)

        # Reconstruction head
        if isinstance(self._target_idx, slice) or len(self._target_idx) > 1:
            x_target = x[:, :, self._target_idx]  # [B,W,out_dim]
        else:
            x_target = x[:, :, [self._target_idx]]  # 강제로 3D로
        recons = self.recon_model(memory, x_target)


        # target_dims를 모델 멤버로 들고 있거나, out_dim에 맞춰 slice
        x_target = x[:, :, self._target_idx] if hasattr(self, "_target_idx") else x
        recons = self.recon_model(memory, x_target)

        return preds, recons


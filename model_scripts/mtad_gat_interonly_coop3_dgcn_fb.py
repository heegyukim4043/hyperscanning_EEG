"""
3-person InterOnly MTAD-GAT with optional:
1) Dynamic Graph Convolution encoder (DGCN)
2) Filter-bank input fusion

Input shapes:
- Standard:          x [B, W, F]
- Filter-bank mode:  x [B, W, BANDS, F]
"""

import math
import torch
import torch.nn as nn

from modules import ConvLayer, GRULayer, Forecasting_Model
from module_transformer import TransformerReconstruction
from module_snn_decoder import RecurrentSNNReconstruction
from dgl_layers_inter3 import DGLInter3GAT

N_CH_PER = 19


def _build_inter_mask(n_features: int, n_ch_per: int) -> torch.Tensor:
    """Inter-brain mask + diagonal self-loop mask: [F, F] boolean."""
    node_idx = torch.arange(n_features)
    person_idx = node_idx // n_ch_per
    inter = person_idx.unsqueeze(1) != person_idx.unsqueeze(0)
    diag = torch.eye(n_features, dtype=torch.bool)
    return inter | diag


class HybridCoopHead3(nn.Module):
    """
    Multi-label cooperation head for pair12 / pair13 / pair23.

    Uses the temporal GRU state as the main detection representation and only a
    compact pair-wise inter-brain attention summary from A_feat. This avoids
    flattening 19x19 attention blocks directly into an MLP.
    """

    def __init__(
        self,
        gru_hid_dim: int,
        n_ch_per: int = 19,
        hidden: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_ch_per = n_ch_per
        self.net = nn.Sequential(
            nn.Linear(gru_hid_dim + 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def _attention_summary(self, A_feat: torch.Tensor) -> torch.Tensor:
        c = self.n_ch_per
        p12 = A_feat[:, :c, c : 2 * c].mean(dim=[1, 2])
        p13 = A_feat[:, :c, 2 * c : 3 * c].mean(dim=[1, 2])
        p23 = A_feat[:, c : 2 * c, 2 * c : 3 * c].mean(dim=[1, 2])
        return torch.stack([p12, p13, p23], dim=1)

    def forward(self, h_last: torch.Tensor, A_feat: torch.Tensor) -> torch.Tensor:
        a_summary = self._attention_summary(A_feat)
        return self.net(torch.cat([h_last, a_summary], dim=1))


class DynamicGraphConvEncoder(nn.Module):
    """
    Dynamic graph convolution over node-time features.
    Builds sample-specific adjacency from node embeddings.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        n_ch_per: int = 19,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.temperature = max(temperature, 1e-6)

        self.q_proj = nn.Linear(window_size, hidden_dim, bias=False)
        self.k_proj = nn.Linear(window_size, hidden_dim, bias=False)
        self.v_proj = nn.Linear(window_size, window_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        mask = _build_inter_mask(n_features=n_features, n_ch_per=n_ch_per)
        self.register_buffer("inter_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor):
        # x: [B, W, F]
        bsz, win, nfeat = x.shape
        if nfeat != self.n_features:
            raise ValueError(f"DGCN expects n_features={self.n_features}, got {nfeat}")
        if win != self.window_size:
            raise ValueError(f"DGCN expects window_size={self.window_size}, got {win}")

        nf = x.transpose(1, 2).contiguous()  # [B, F, W]
        q = self.q_proj(nf)                  # [B, F, D]
        k = self.k_proj(nf)                  # [B, F, D]

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        logits = logits / self.temperature

        mask = self.inter_mask.unsqueeze(0)  # [1, F, F]
        logits = logits.masked_fill(~mask, float("-inf"))

        a_dyn = torch.softmax(logits, dim=-1)  # [B, F, F]
        a_dyn = self.dropout(a_dyn)

        out_nf = torch.matmul(a_dyn, nf)       # [B, F, W]
        out_nf = self.v_proj(out_nf)           # [B, F, W]
        h_dyn = out_nf.transpose(1, 2).contiguous()  # [B, W, F]
        return h_dyn, a_dyn


class MTAD_GAT_InterOnly_Coop3_DGCN_FB(nn.Module):
    """
    InterOnly Coop3 model with optional DGCN and filter-bank support.

    Returns:
    - preds:       [B, out_dim]
    - recons:      [B, W, out_dim]
    - coop_logits: [B, 3]
    - A_feat:      [B, F, F]
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        out_dim: int,
        kernel_size: int = 7,
        use_gatv2: bool = True,
        gat_heads_feat: int = 2,
        gru_n_layers: int = 1,
        gru_hid_dim: int = 96,
        fc_n_layers: int = 1,
        fc_hid_dim: int = 96,
        recon_d_model: int = 64,
        recon_nhead: int = 4,
        recon_num_layers: int = 1,
        recon_dim_ff: int = 128,
        dropout: float = 0.25,
        alpha: float = 0.2,
        n_ch_per: int = 19,
        coop_hidden: int = 64,
        use_dgcn: bool = False,
        dgcn_hidden_dim: int = 64,
        dgcn_dropout: float = 0.1,
        dgcn_temperature: float = 1.0,
        use_filterbank: bool = False,
        fb_num_bands: int = 5,
        fb_fusion: str = "mean",  # mean | attn
        decoder_type: str = "transformer",  # transformer | snn_rnn
        snn_hidden_dim: int = 128,
        snn_num_layers: int = 1,
        snn_dropout: float = 0.10,
        snn_threshold: float = 1.0,
        snn_surrogate_beta: float = 10.0,
        snn_learnable_decay: bool = True,
        sync_prior_enabled: bool = False,
        sync_prior_matrix: torch.Tensor = None,
        sync_prior_lambda: float = 0.0,
        sync_prior_mix: float = 1.0,
    ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.out_dim = out_dim
        self.n_ch_per = n_ch_per
        self.use_dgcn = use_dgcn
        self.use_filterbank = use_filterbank
        self.fb_num_bands = max(int(fb_num_bands), 1)
        self.fb_fusion = fb_fusion
        self.decoder_type = decoder_type
        self.sync_prior_enabled = bool(sync_prior_enabled)
        self.sync_prior_lambda = float(sync_prior_lambda)
        self.sync_prior_mix = float(max(0.0, min(1.0, sync_prior_mix)))

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = DGLInter3GAT(
            n_features=n_features,
            window_size=window_size,
            n_ch_per=n_ch_per,
            out_window_size=window_size,
            num_heads=gat_heads_feat,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
        )

        if self.use_dgcn:
            self.dynamic_encoder = DynamicGraphConvEncoder(
                n_features=n_features,
                window_size=window_size,
                n_ch_per=n_ch_per,
                hidden_dim=dgcn_hidden_dim,
                dropout=dgcn_dropout,
                temperature=dgcn_temperature,
            )
            self.dgcn_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        else:
            self.dynamic_encoder = None
            self.register_parameter("dgcn_gate", None)

        if self.use_filterbank and self.fb_fusion == "attn":
            self.fb_weight_bias = nn.Parameter(torch.zeros(self.fb_num_bands))
            self.fb_temp = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("fb_weight_bias", None)
            self.register_parameter("fb_temp", None)

        sync_mask = _build_inter_mask(n_features=n_features, n_ch_per=n_ch_per).float()
        self.register_buffer("sync_prior_mask", sync_mask, persistent=False)
        if sync_prior_matrix is not None:
            prior = torch.as_tensor(sync_prior_matrix, dtype=torch.float32)
            if prior.shape != (n_features, n_features):
                raise ValueError(
                    f"sync_prior_matrix shape mismatch: expected {(n_features, n_features)}, got {tuple(prior.shape)}"
                )
            self.register_buffer("sync_prior_matrix", prior, persistent=False)
        else:
            self.register_buffer("sync_prior_matrix", None, persistent=False)

        self.gru = GRULayer(2 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, fc_hid_dim, out_dim, fc_n_layers, dropout)
        if decoder_type == "transformer":
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
        elif decoder_type in ("snn_rnn", "snn"):
            self.recon_model = RecurrentSNNReconstruction(
                window_size=window_size,
                enc_dim=gru_hid_dim,
                out_dim=out_dim,
                hidden_dim=snn_hidden_dim,
                num_layers=snn_num_layers,
                dropout=snn_dropout,
                threshold=snn_threshold,
                surrogate_beta=snn_surrogate_beta,
                learnable_decay=snn_learnable_decay,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.coop_head = HybridCoopHead3(
            gru_hid_dim=gru_hid_dim,
            n_ch_per=n_ch_per,
            hidden=coop_hidden,
            dropout=dropout,
        )

        self.last_band_weights = None
        self.last_dgcn_gate = None

    def _apply_sync_prior(self, a_feat: torch.Tensor) -> torch.Tensor:
        if (not self.sync_prior_enabled) or (self.sync_prior_matrix is None) or self.sync_prior_lambda == 0.0:
            return a_feat
        eps = 1e-8
        prior = self.sync_prior_matrix.to(device=a_feat.device, dtype=a_feat.dtype)
        mask = self.sync_prior_mask.to(device=a_feat.device, dtype=torch.bool)

        logits = torch.log(torch.clamp(a_feat, min=eps)) + self.sync_prior_lambda * prior.unsqueeze(0)
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        a_prior = torch.softmax(logits, dim=-1)
        if self.sync_prior_mix >= 1.0:
            return a_prior
        return (1.0 - self.sync_prior_mix) * a_feat + self.sync_prior_mix * a_prior

    def _band_weights(self, x_fb: torch.Tensor) -> torch.Tensor:
        # x_fb: [B, W, BANDS, F]
        bsz, _, bands, _ = x_fb.shape
        if self.fb_fusion == "mean":
            return x_fb.new_full((bsz, bands), 1.0 / bands)

        # attention-like global band weighting
        energy = x_fb.abs().mean(dim=(1, 3))      # [B, BANDS]
        logits = energy * torch.clamp(self.fb_temp, min=1e-3)
        if self.fb_weight_bias is not None:
            logits = logits + self.fb_weight_bias[None, :bands]
        return torch.softmax(logits, dim=1)

    def _encode_with_filterbank(self, x_fb: torch.Tensor):
        # x_fb: [B, W, BANDS, F]
        bsz, win, bands, nfeat = x_fb.shape
        if nfeat != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {nfeat}")
        if win != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {win}")

        w_band = self._band_weights(x_fb)  # [B, BANDS]
        self.last_band_weights = w_band.detach()

        h_list, a_list = [], []
        for bi in range(bands):
            xb = x_fb[:, :, bi, :]
            hb, ab = self.feature_gat(xb, return_attn=True)
            h_list.append(hb)
            a_list.append(ab)

        h_stack = torch.stack(h_list, dim=1)  # [B, BANDS, W, F]
        a_stack = torch.stack(a_list, dim=1)  # [B, BANDS, F, F]
        wb_h = w_band[:, :, None, None]      # [B, BANDS, 1, 1]
        h_feat = (h_stack * wb_h).sum(dim=1)
        a_feat = (a_stack * wb_h).sum(dim=1)

        # fused signal used by GRU/reconstruction branch
        wb_x = w_band[:, None, :, None]     # [B, 1, BANDS, 1]
        x_main = (x_fb * wb_x).sum(dim=2)   # [B, W, F]
        return x_main, h_feat, a_feat

    def forward(self, x: torch.Tensor):
        # x: [B, W, F] or [B, W, BANDS, F]
        if x.dim() == 3:
            x_main = x
            h_gat, a_gat = self.feature_gat(x_main, return_attn=True)
            self.last_band_weights = None
        elif x.dim() == 4:
            if not self.use_filterbank:
                raise ValueError("Received filter-bank input [B,W,BANDS,F] but use_filterbank=False")
            x_main, h_gat, a_gat = self._encode_with_filterbank(x)
        else:
            raise ValueError(f"Unsupported input rank: {x.dim()}")

        if self.use_dgcn:
            h_dyn, a_dyn = self.dynamic_encoder(x_main)
            gate = torch.sigmoid(self.dgcn_gate)
            self.last_dgcn_gate = float(gate.detach().cpu())
            h_feat = gate * h_gat + (1.0 - gate) * h_dyn
            a_feat = gate * a_gat + (1.0 - gate) * a_dyn
        else:
            self.last_dgcn_gate = None
            h_feat, a_feat = h_gat, a_gat

        a_feat = self._apply_sync_prior(a_feat)

        h_cat = torch.cat([x_main, h_feat], dim=2)  # [B, W, 2F]
        memory, _ = self.gru(h_cat)
        if memory.dim() == 3 and memory.shape[0] == x_main.shape[1] and memory.shape[1] == x_main.shape[0]:
            memory = memory.transpose(0, 1).contiguous()

        h_last = memory[:, -1, :]
        preds = self.forecasting_model(h_last)

        x_target = x_main[:, :, : self.out_dim]
        recons = self.recon_model(memory, x_target)
        coop_logits = self.coop_head(h_last, a_feat)
        return preds, recons, coop_logits, a_feat

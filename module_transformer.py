import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TransformerReconstruction(nn.Module):
    """
    GRU memory(encoder output) + teacher forcing target로
    Transformer Decoder를 통해 [B,W,out_dim] reconstruction 생성
    """
    def __init__(
        self,
        window_size: int,
        enc_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.out_dim = out_dim
        self.d_model = d_model

        self.proj_enc = nn.Linear(enc_dim, d_model)
        self.proj_dec_in = nn.Linear(out_dim, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=window_size)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # torch 2.x에서는 지원됨
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

        self.start_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, memory: torch.Tensor, x_target: torch.Tensor):
        """
        memory:   [B, W, enc_dim]
        x_target: [B, W, out_dim]  (teacher forcing target)
        """
        B, W, _ = x_target.shape
        device = x_target.device

        # encoder memory
        mem = self.proj_enc(memory)          # [B,W,d_model]
        mem = self.posenc(mem)

        # decoder input: shift-right
        start = self.start_token.expand(B, 1, self.out_dim).to(device)
        tgt = torch.cat([start, x_target[:, :-1, :]], dim=1)  # [B,W,out_dim]
        tgt = self.proj_dec_in(tgt)                           # [B,W,d_model]
        tgt = self.posenc(tgt)

        # causal mask: [W,W] with True where masked
        tgt_mask = torch.triu(torch.ones(W, W, device=device), diagonal=1).bool()

        out = self.decoder(tgt, mem, tgt_mask=tgt_mask)       # [B,W,d_model]
        recons = self.out_proj(out)                            # [B,W,out_dim]
        return recons

import torch
import torch.nn as nn

from dgl_layers_full import DGLFeatureGAT, DGLTemporalGAT


class HyperscanMTADGAT_PairHeads(nn.Module):
    """
    Hyperscanning multi-label model with pairwise heads.

    Input:
      x: [B, W, 57]  where 57 = 3 * 19  (P1 19ch | P2 19ch | P3 19ch)

    Output:
      logits: [B, 3]  corresponding to [i(1-2), j(1-3), k(2-3)]
    """

    def __init__(
        self,
        n_person: int = 3,
        ch_per_person: int = 19,
        window_size: int = 150,
        gat_heads: int = 2,
        dropout: float = 0.25,
        alpha: float = 0.2,
        use_gatv2: bool = True,
        fuse_mode: str = "concat",     # "concat" or "sum"
        seq_encoder: str = "gru",      # "gru" or "transformer-lite"
        person_emb_dim: int = 128,     # person embedding size
        gru_hid_dim: int = 150,
        gru_n_layers: int = 1,
        pair_mlp_dim: int = 128,
        return_aux: bool = False,
    ):
        super().__init__()
        self.n_person = n_person
        self.ch_per_person = ch_per_person
        self.F = n_person * ch_per_person
        self.W = window_size
        self.fuse_mode = fuse_mode
        self.seq_encoder = seq_encoder
        self.return_aux = return_aux

        # --- Graph backbone (Option B) ---
        self.feat_gat = DGLFeatureGAT(
            n_features=self.F,
            window_size=self.W,
            out_window_size=self.W,
            num_heads=gat_heads,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
        )
        self.time_gat = DGLTemporalGAT(
            n_features=self.F,
            window_size=self.W,
            out_features=self.F,
            num_heads=gat_heads,
            dropout=dropout,
            alpha=alpha,
            use_gatv2=use_gatv2,
            self_loop=True,
            band_k=10,
        )

        if fuse_mode == "concat":
            self.fuse_proj = nn.Sequential(
                nn.Linear(2 * self.F, self.F),
                nn.Dropout(dropout),
            )
        elif fuse_mode == "sum":
            self.fuse_proj = nn.Identity()
        else:
            raise ValueError("fuse_mode must be 'concat' or 'sum'")

        # --- Temporal encoder to get window-level representation per channel ---
        # We want a compact representation of each channel across time, then pool into person embedding.
        if self.seq_encoder == "gru":
            self.temporal = nn.GRU(
                input_size=self.F,
                hidden_size=gru_hid_dim,
                num_layers=gru_n_layers,
                batch_first=True,
                dropout=0.0 if gru_n_layers == 1 else dropout,
                bidirectional=False,
            )
            win_dim = gru_hid_dim
        elif self.seq_encoder == "transformer-lite":
            # lightweight transformer encoder over time (works as window summarizer)
            d_model = 128
            nhead = 4
            num_layers = 2
            self.in_proj = nn.Linear(self.F, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            win_dim = d_model
        else:
            raise ValueError("seq_encoder must be 'gru' or 'transformer-lite'")

        # --- Person embedding projector ---
        # We will build person embedding from window summary, using person-specific channel pooling.
        self.person_proj = nn.Sequential(
            nn.Linear(win_dim, person_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.person_token = nn.Parameter(torch.zeros(self.n_person, person_emb_dim))
        nn.init.normal_(self.person_token, std=0.02)

        # --- Pair heads: (12), (13), (23) ---
        # Pair feature = [p, q, |p-q|, p*q] (common in metric learning / interaction modeling)
        pair_in_dim = 4 * person_emb_dim

        def make_pair_head():
            return nn.Sequential(
                nn.Linear(pair_in_dim, pair_mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pair_mlp_dim, 1),  # single logit
            )

        self.head_12 = make_pair_head()
        self.head_13 = make_pair_head()
        self.head_23 = make_pair_head()

    def _split_person_blocks(self, h: torch.Tensor):
        """
        h: [B, D] window-level vector
        Since the temporal encoder yields a window-level embedding (per window), we then create person embeddings.
        Here D = win_dim.
        """
        # This function is not used; person embeddings are created from window summary, not per-channel.
        raise NotImplementedError

    def _person_embeddings_from_window(self, win_vec: torch.Tensor):
        """
        win_vec: [B, D]  (window summary)
        returns list: [p1, p2, p3], each [B, E]
        """
        base = self.person_proj(win_vec)  # [B, E]

        persons = []
        for pid in range(self.n_person):
            persons.append(base + self.person_token[pid])  # token is on same device as model
        return persons

    @staticmethod
    def _pair_feat(p: torch.Tensor, q: torch.Tensor):
        # p,q: [B,E]
        return torch.cat([p, q, torch.abs(p - q), p * q], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        x: [B,W,F=57]
        returns:
          logits: [B,3]
          (optional) aux dict
        """
        B, W, F = x.shape
        assert F == self.F and W == self.W, f"Expected x [B,{self.W},{self.F}] got {x.shape}"

        # Graph backbone
        h_feat = self.feat_gat(x)  # [B,W,F]
        h_time = self.time_gat(x)  # [B,W,F]

        if self.fuse_mode == "concat":
            h = torch.cat([h_feat, h_time], dim=-1)  # [B,W,2F]
            h = self.fuse_proj(h)                    # [B,W,F]
        else:
            h = h_feat + h_time                      # [B,W,F]

        # Temporal summarization of the window
        if self.seq_encoder == "gru":
            _, h_last = self.temporal(h)             # h_last: [L,B,H]
            win_vec = h_last[-1]                     # [B,H]
        else:
            z = self.temporal(self.in_proj(h))       # [B,W,D]
            win_vec = z[:, -1, :]                    # last-step summary (can be mean pooling too)

        # Person embeddings (p1, p2, p3): each [B,E]
        p1, p2, p3 = self._person_embeddings_from_window(win_vec)

        # Pair logits
        logit_12 = self.head_12(self._pair_feat(p1, p2)).squeeze(-1)  # [B]
        logit_13 = self.head_13(self._pair_feat(p1, p3)).squeeze(-1)
        logit_23 = self.head_23(self._pair_feat(p2, p3)).squeeze(-1)

        logits = torch.stack([logit_12, logit_13, logit_23], dim=-1)  # [B,3]

        if self.return_aux:
            aux = {
                "p1": p1, "p2": p2, "p3": p3,
                "win_vec": win_vec,
            }
            return logits, aux

        return logits

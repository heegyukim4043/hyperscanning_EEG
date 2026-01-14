# dgl_layers_full.py
import dgl
import torch
import torch.nn as nn

try:
    from dgl.nn.pytorch import GATConv, GATv2Conv
except Exception as e:
    raise ImportError("DGL pytorch backend가 필요합니다. `pip install dgl` 확인") from e


def _complete_graph(num_nodes: int, self_loop: bool = True, device=None):
    """
    Fully-connected directed graph.
    - self_loop=True면 (i->i) 포함
    """
    src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
    if not self_loop:
        mask = src != dst
        src, dst = src[mask], dst[mask]
    g = dgl.graph((src, dst), num_nodes=num_nodes, device=device)
    return g


def _edge_attn_to_adj(attn, src0, dst0, B: int, N: int, num_heads: int):
    """
    attn: (B*E, num_heads, 1) 또는 (B*E, num_heads)
    src0, dst0: base_graph의 edge index (E,)
    return: A (B, num_heads, N, N) where A[b,h,src,dst] = attn value
    """
    if attn.dim() == 3 and attn.size(-1) == 1:
        attn = attn.squeeze(-1)  # (B*E, H)
    elif attn.dim() != 2:
        raise ValueError(f"Unexpected attn shape: {tuple(attn.shape)}")

    E = src0.numel()
    attn = attn.view(B, E, num_heads).permute(0, 2, 1).contiguous()  # (B,H,E)

    idx = (src0 * N + dst0).long()  # (E,)
    Aflat = attn.new_zeros((B, num_heads, N * N))
    Aflat[:, :, idx] = attn
    A = Aflat.view(B, num_heads, N, N)
    return A


class DGLFeatureGAT(nn.Module):
    """
    Feature-graph GAT:
    - nodes: features (F)
    - node feature dim: window size (W)  -> 각 feature의 time-series 벡터
    input:  x [B, W, F]
    output: h_feat [B, W, F]
    """
    def __init__(
        self,
        n_features: int,
        window_size: int,
        out_window_size: int = None,
        num_heads: int = 2,
        dropout: float = 0.2,
        alpha: float = 0.2,
        use_gatv2: bool = True,
        self_loop: bool = True,
    ):
        super().__init__()
        self.F = n_features
        self.W = window_size
        self.outW = out_window_size if out_window_size is not None else window_size
        self.num_heads = num_heads
        self.self_loop = self_loop
        self.use_gatv2 = use_gatv2

        Conv = GATv2Conv if use_gatv2 else GATConv

        # in_dim = W, out_dim = outW
        # DGL GATConv의 out_feats는 head당 output dim
        self.gat = Conv(
            in_feats=self.W,
            out_feats=self.outW,
            num_heads=self.num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            negative_slope=alpha,
            allow_zero_in_degree=True,
        )

        # head aggregation 후 F차원으로 되돌리는 projection
        # (B,F,outW) -> (B,F,outW) -> transpose -> (B,outW,F)
        self.head_aggr = "mean"  # mean or flatten
        if self.head_aggr == "flatten":
            self.proj = nn.Linear(self.num_heads * self.outW, self.outW)
        else:
            self.proj = nn.Identity()

        self._base_graph = None
        self._batched_graph_cache = {}  # key=(B, device_str)

    def _get_batched_graph(self, batch_size: int, device):
        key = (batch_size, str(device))
        if key in self._batched_graph_cache:
            return self._batched_graph_cache[key]

        if self._base_graph is None or self._base_graph.device != device:
            self._base_graph = _complete_graph(self.F, self_loop=self.self_loop, device=device)

        # B개 그래프 배치
        g = dgl.batch([self._base_graph] * batch_size)
        self._batched_graph_cache[key] = g
        return g

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, W, F = x.shape
        device = x.device
        g = self._get_batched_graph(B, device)

        nf = x.transpose(1, 2).contiguous().view(B * F, W)  # [B*F, W]

        if return_attn:
            out, attn = self.gat(g, nf, get_attention=True)
            if attn.dim() == 3:  # [E, heads, 1] -> [E, heads]
                attn = attn.squeeze(-1)
        else:
            out = self.gat(g, nf)
            attn = None

        out = out.mean(dim=1)  # [B*F, outW]
        h_feat = out.view(B, F, self.outW).transpose(1, 2).contiguous()

        if not return_attn:
            return h_feat

        # ---- (B,F,F) adjacency로 변환 (heads 평균) ----
        E0 = self.F * self.F if self.self_loop else self.F * (self.F - 1)
        attn = attn.view(B, E0, self.num_heads).mean(dim=2)  # [B, E0]

        src, dst = self._base_graph.edges()
        A = torch.zeros(B, self.F, self.F, device=device)
        A[:, src, dst] = attn
        return h_feat, A

class DGLTemporalGAT(nn.Module):
    """
    Temporal-graph GAT:
    - nodes: time steps (W)
    - node feature dim: features (F) -> 각 time의 feature 벡터
    input:  x [B, W, F]
    output: h_temp [B, W, F]
    """
    def __init__(
        self,
        n_features: int,
        window_size: int,
        out_features: int = None,
        num_heads: int = 2,
        dropout: float = 0.2,
        alpha: float = 0.2,
        use_gatv2: bool = True,
        self_loop: bool = True,
    ):
        super().__init__()
        self.F = n_features
        self.W = window_size
        self.outF = out_features if out_features is not None else n_features
        self.num_heads = num_heads
        self.self_loop = self_loop
        self.use_gatv2 = use_gatv2

        Conv = GATv2Conv if use_gatv2 else GATConv

        self.gat = Conv(
            in_feats=self.F,
            out_feats=self.outF,
            num_heads=self.num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            negative_slope=alpha,
            allow_zero_in_degree=True,
        )

        self.head_aggr = "mean"
        if self.head_aggr == "flatten":
            self.proj = nn.Linear(self.num_heads * self.outF, self.outF)
        else:
            self.proj = nn.Identity()

        self._base_graph = None
        self._batched_graph_cache = {}

    def _get_batched_graph(self, batch_size: int, device):
        key = (batch_size, str(device))
        if key in self._batched_graph_cache:
            return self._batched_graph_cache[key]

        if self._base_graph is None or self._base_graph.device != device:
            self._base_graph = _complete_graph(self.W, self_loop=self.self_loop, device=device)

        g = dgl.batch([self._base_graph] * batch_size)
        self._batched_graph_cache[key] = g
        return g

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, W, F = x.shape
        device = x.device
        g = self._get_batched_graph(B, device)

        nf = x.contiguous().view(B * W, F)  # [B*W, F]

        if return_attn:
            out, attn = self.gat(g, nf, get_attention=True)
            if attn.dim() == 3:
                attn = attn.squeeze(-1)
        else:
            out = self.gat(g, nf)
            attn = None

        out = out.mean(dim=1)
        h_temp = out.view(B, W, self.outF)

        if not return_attn:
            return h_temp

        E0 = self.W * self.W if self.self_loop else self.W * (self.W - 1)
        attn = attn.view(B, E0, self.num_heads).mean(dim=2)  # [B,E0]

        src, dst = self._base_graph.edges()
        A = torch.zeros(B, self.W, self.W, device=device)
        A[:, src, dst] = attn
        return h_temp, A

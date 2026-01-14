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


def _band_graph(num_nodes: int, k: int, self_loop: bool = True, device=None):
    """
    Band directed graph with edges i->j if |i-j| <= k.
    - self_loop=True면 i->i 포함 (k와 무관)
    """
    if k < 0:
        raise ValueError("k must be >= 0 for band graph")

    idx = torch.arange(num_nodes, device=device)
    src_list = []
    dst_list = []

    # self loop
    if self_loop:
        src_list.append(idx)
        dst_list.append(idx)

    for d in range(1, k + 1):
        # i -> i+d
        src_fwd = idx[:-d]
        dst_fwd = idx[d:]
        # i -> i-d
        src_bwd = idx[d:]
        dst_bwd = idx[:-d]
        src_list.extend([src_fwd, src_bwd])
        dst_list.extend([dst_fwd, dst_bwd])

    src = torch.cat(src_list, dim=0) if len(src_list) > 0 else torch.empty(0, dtype=torch.int64, device=device)
    dst = torch.cat(dst_list, dim=0) if len(dst_list) > 0 else torch.empty(0, dtype=torch.int64, device=device)

    g = dgl.graph((src, dst), num_nodes=num_nodes, device=device)
    return g


def _edge_attn_to_dense_adj(g: dgl.DGLGraph, attn: torch.Tensor, num_nodes: int, batch_size: int) -> torch.Tensor:
    """
    Convert edge attention (E_total, heads, 1) or (E_total, heads) to dense adjacency (B, N, N).
    We average across heads.
    """
    if attn.dim() == 3:
        attn = attn.squeeze(-1)  # [E_total, heads]
    attn_mean = attn.mean(dim=1)  # [E_total]

    src, dst = g.edges()
    gid = (src // num_nodes).long()
    src_local = (src % num_nodes).long()
    dst_local = (dst % num_nodes).long()

    A = torch.zeros(batch_size, num_nodes, num_nodes, device=attn.device, dtype=attn_mean.dtype)
    A[gid, src_local, dst_local] = attn_mean
    return A


class DGLFeatureGAT(nn.Module):
    """
    Feature-graph GAT:
    - nodes: features (F)
    - node feature dim: window size (W)
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

        self.gat = Conv(
            in_feats=self.W,
            out_feats=self.outW,
            num_heads=self.num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            negative_slope=alpha,
            allow_zero_in_degree=True,
        )

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

        g = dgl.batch([self._base_graph] * batch_size)
        self._batched_graph_cache[key] = g
        return g

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, W, F = x.shape
        assert F == self.F and W == self.W, f"Expected x [B,{self.W},{self.F}] got {x.shape}"

        device = x.device
        g = self._get_batched_graph(B, device)

        nf = x.transpose(1, 2).contiguous()        # [B,F,W]
        nf = nf.view(B * F, W)                     # [B*F, W]

        if return_attn:
            out, attn = self.gat(g, nf, get_attention=True)  # out: [B*F, heads, outW]
        else:
            out = self.gat(g, nf)
            attn = None

        if self.head_aggr == "flatten":
            out = out.reshape(B * F, self.num_heads * self.outW)
            out = self.proj(out)
        else:
            out = out.mean(dim=1)                  # [B*F, outW]

        out = out.view(B, F, self.outW)            # [B,F,outW]
        h_feat = out.transpose(1, 2).contiguous()  # [B,outW,F] == [B,W,F]

        if return_attn:
            A_feat = _edge_attn_to_dense_adj(g, attn, num_nodes=F, batch_size=B)  # [B,F,F]
            return h_feat, A_feat

        return h_feat


class DGLTemporalGAT(nn.Module):
    """
    Temporal-graph GAT:
    - nodes: time steps (W)
    - node feature dim: features (F)
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
        band_k: int = None,   # None -> complete graph, int -> band graph (±k)
    ):
        super().__init__()
        self.F = n_features
        self.W = window_size
        self.outF = out_features if out_features is not None else n_features
        self.num_heads = num_heads
        self.self_loop = self_loop
        self.use_gatv2 = use_gatv2
        self.band_k = band_k

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
            if self.band_k is None:
                self._base_graph = _complete_graph(self.W, self_loop=self.self_loop, device=device)
            else:
                self._base_graph = _band_graph(self.W, k=int(self.band_k), self_loop=self.self_loop, device=device)

        g = dgl.batch([self._base_graph] * batch_size)
        self._batched_graph_cache[key] = g
        return g

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, W, F = x.shape
        assert F == self.F and W == self.W, f"Expected x [B,{self.W},{self.F}] got {x.shape}"

        device = x.device
        g = self._get_batched_graph(B, device)

        nf = x.contiguous().view(B * W, F)         # [B*W, F]

        if return_attn:
            out, attn = self.gat(g, nf, get_attention=True)  # [B*W, heads, outF]
        else:
            out = self.gat(g, nf)
            attn = None

        if self.head_aggr == "flatten":
            out = out.reshape(B * W, self.num_heads * self.outF)
            out = self.proj(out)
        else:
            out = out.mean(dim=1)                  # [B*W, outF]

        h_temp = out.view(B, W, self.outF)         # [B,W,F]

        if return_attn:
            A_time = _edge_attn_to_dense_adj(g, attn, num_nodes=W, batch_size=B)  # [B,W,W]
            return h_temp, A_time

        return h_temp

# dgl_layers_inter3.py
"""
3-Person Inter-brain GAT
=========================
3명 피험자(p0, p1, p2) 간 inter-brain edges만 사용하는 그래프.

노드:
  p0: 0  ~ 18   (19채널)
  p1: 19 ~ 37
  p2: 38 ~ 56

엣지 (inter-brain only, 방향 그래프):
  p0 <-> p1  (pair12)
  p0 <-> p2  (pair13)
  p1 <-> p2  (pair23)
  + optional self-loop

A_feat 블록 (57×57):
  [0:19,  19:38] = pair12 attention
  [0:19,  38:57] = pair13 attention
  [19:38, 38:57] = pair23 attention
"""

import dgl
import torch
import torch.nn as nn

try:
    from dgl.nn.pytorch import GATv2Conv, GATConv
except Exception as e:
    raise ImportError("DGL pytorch backend 필요: pip install dgl") from e


def _inter3_graph(n_ch_per: int, self_loop: bool = True, device=None):
    """
    3-person inter-brain directed graph.
    Total nodes = 3 * n_ch_per (e.g. 57).
    Connects all pairs: p0<->p1, p0<->p2, p1<->p2.
    """
    C = n_ch_per
    p0 = torch.arange(0,     C,     device=device)
    p1 = torch.arange(C,     2 * C, device=device)
    p2 = torch.arange(2 * C, 3 * C, device=device)

    srcs, dsts = [], []
    for pa, pb in [(p0, p1), (p0, p2), (p1, p2)]:
        # pa -> pb
        srcs.append(pa.repeat_interleave(C))
        dsts.append(pb.repeat(C))
        # pb -> pa
        srcs.append(pb.repeat_interleave(C))
        dsts.append(pa.repeat(C))

    src = torch.cat(srcs)
    dst = torch.cat(dsts)

    if self_loop:
        all_nodes = torch.arange(3 * C, device=device)
        src = torch.cat([src, all_nodes])
        dst = torch.cat([dst, all_nodes])

    return dgl.graph((src, dst), num_nodes=3 * C, device=device)


def _edge_attn_to_dense_adj(g, attn, num_nodes, batch_size):
    if attn.dim() == 3:
        attn = attn.squeeze(-1)
    attn_mean = attn.mean(dim=1)

    src, dst   = g.edges()
    gid        = (src // num_nodes).long()
    src_local  = (src %  num_nodes).long()
    dst_local  = (dst %  num_nodes).long()

    A = torch.zeros(batch_size, num_nodes, num_nodes,
                    device=attn.device, dtype=attn_mean.dtype)
    A[gid, src_local, dst_local] = attn_mean
    return A


class DGLInter3GAT(nn.Module):
    """
    3-person inter-brain GAT.

    input : x  [B, W, F]  F = 3 * n_ch_per = 57
    output: h  [B, W, F]
            A  [B, F, F]   (non-zero in inter-brain blocks + diagonal)
    """
    def __init__(
        self,
        n_features:      int,         # 3 * n_ch_per
        window_size:     int,
        n_ch_per:        int = 19,
        out_window_size: int = None,
        num_heads:       int = 2,
        dropout:         float = 0.2,
        alpha:           float = 0.2,
        use_gatv2:       bool = True,
        self_loop:       bool = True,
    ):
        super().__init__()
        self.F        = n_features
        self.W        = window_size
        self.outW     = out_window_size or window_size
        self.n_ch_per = n_ch_per
        self.num_heads = num_heads
        self.self_loop = self_loop

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

        self._base_graph = None
        self._batched_graph_cache = {}

    def _get_batched_graph(self, batch_size, device):
        key = (batch_size, str(device))
        if key not in self._batched_graph_cache:
            if self._base_graph is None:
                self._base_graph = _inter3_graph(
                    self.n_ch_per, self.self_loop, device=device
                )
            self._batched_graph_cache[key] = dgl.batch(
                [self._base_graph] * batch_size
            )
        return self._batched_graph_cache[key]

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, W, F = x.shape
        device  = x.device

        g  = self._get_batched_graph(B, device)
        nf = x.transpose(1, 2).contiguous().view(B * F, W)

        if return_attn:
            out, attn = self.gat(g, nf, get_attention=True)
        else:
            out  = self.gat(g, nf)
            attn = None

        out = out.mean(dim=1).view(B, F, self.outW)
        h   = out.transpose(1, 2).contiguous()

        if return_attn:
            A = _edge_attn_to_dense_adj(g, attn, num_nodes=F, batch_size=B)
            return h, A
        return h

import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv


def _build_fully_connected_graph(num_nodes: int, device):
    # self-loop 포함 fully connected
    src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
    g = dgl.graph((src, dst), num_nodes=num_nodes, device=device)
    return g


class DGLFeatureGAT(nn.Module):
    """
    Feature graph: nodes=F, each node feature is a time-series vector length W.
    Input:  x [B,W,F]
    Output: h_feat [B,W,F]  (same shape to concatenate)
    """
    def __init__(self, n_features: int, window_size: int,
                 num_heads: int = 4, head_dim: int = 8,
                 feat_drop: float = 0.1, attn_drop: float = 0.1):
        super().__init__()
        self.F = n_features
        self.W = window_size
        self.num_heads = num_heads
        self.out_dim = num_heads * head_dim

        # node feature dim = W
        self.gat = GATConv(
            in_feats=window_size,
            out_feats=head_dim,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            allow_zero_in_degree=True,
        )
        # project back to W (so that we can output [B,W,F])
        self.proj_back = nn.Linear(self.out_dim, window_size)

    def forward(self, x: torch.Tensor):
        B, W, F = x.shape
        assert F == self.F and W == self.W, f"Expected [B,{self.W},{self.F}], got {x.shape}"
        device = x.device

        g = _build_fully_connected_graph(self.F, device=device)

        # Node feature per feature-node: [B, F, W]
        node_feat = x.transpose(1, 2)  # [B,F,W]

        # DGL GATConv expects [N, in_dim] (for 1 graph). We run B graphs by flattening:
        # batch graphs: use dgl.batch
        graphs = [g] * B
        bg = dgl.batch(graphs)

        # Flatten node features: [B*F, W]
        h0 = node_feat.reshape(B * self.F, self.W)

        h = self.gat(bg, h0)                   # [B*F, num_heads, head_dim]
        h = h.reshape(B * self.F, -1)          # [B*F, out_dim]
        h = self.proj_back(h)                  # [B*F, W]
        h = h.reshape(B, self.F, self.W)       # [B,F,W]
        h = h.transpose(1, 2)                  # [B,W,F]
        return h


class DGLTemporalGAT(nn.Module):
    """
    Temporal graph: nodes=W, each node feature is a feature vector length F.
    Input:  x [B,W,F]
    Output: h_time [B,W,F]
    """
    def __init__(self, n_features: int, window_size: int,
                 num_heads: int = 4, head_dim: int = 8,
                 feat_drop: float = 0.1, attn_drop: float = 0.1):
        super().__init__()
        self.F = n_features
        self.W = window_size
        self.num_heads = num_heads
        self.out_dim = num_heads * head_dim

        # node feature dim = F
        self.gat = GATConv(
            in_feats=n_features,
            out_feats=head_dim,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            allow_zero_in_degree=True,
        )
        self.proj_back = nn.Linear(self.out_dim, n_features)

    def forward(self, x: torch.Tensor):
        B, W, F = x.shape
        assert F == self.F and W == self.W, f"Expected [B,{self.W},{self.F}], got {x.shape}"
        device = x.device

        g = _build_fully_connected_graph(self.W, device=device)

        # Node feature per time-node: [B, W, F] already
        graphs = [g] * B
        bg = dgl.batch(graphs)

        h0 = x.reshape(B * self.W, self.F)     # [B*W, F]
        h = self.gat(bg, h0)                   # [B*W, num_heads, head_dim]
        h = h.reshape(B * self.W, -1)          # [B*W, out_dim]
        h = self.proj_back(h)                  # [B*W, F]
        h = h.reshape(B, self.W, self.F)       # [B,W,F]
        return h

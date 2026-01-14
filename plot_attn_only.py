import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

import networkx as nx

# ---- 위에서 고친 plot_topk_digraph 그대로 복사 ----
def plot_topk_digraph(A, out_png, title: str, topk: int = 250, seed: int = 0):
    import numpy as np
    from pathlib import Path

    out_png = Path(out_png)

    A = A.detach().float().cpu().numpy().copy()
    np.fill_diagonal(A, 0.0)

    N = A.shape[0]
    flat = A.ravel()
    k = min(topk, flat.size)

    idx = np.argsort(flat)[::-1][:k]
    src, dst = np.unravel_index(idx, (N, N))

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for s, d in zip(src, dst):
        w = float(A[s, d])
        if w > 0:
            G.add_edge(int(s), int(d), weight=w)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.axis("off")

    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_nodes(G, pos, node_size=25, ax=ax)

    if G.number_of_edges() == 0:
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->", arrowsize=6,
        edge_color=weights, edge_cmap=plt.cm.Reds,
        width=1.0, alpha=0.6,
        ax=ax
    )

    vmax = max(1e-6, float(weights.max()))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])

    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_heatmap(A, out_png, title: str):
    A = A.detach().float().cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(A, aspect="auto")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_dir", required=True, help="attn 폴더 경로")
    ap.add_argument("--topk_feat", type=int, default=250)
    ap.add_argument("--topk_time", type=int, default=600)
    args = ap.parse_args()

    attn_dir = Path(args.attn_dir)

    A0f = torch.from_numpy(np.load(attn_dir / "A_feat_label0.npy"))
    A1f = torch.from_numpy(np.load(attn_dir / "A_feat_label1.npy"))
    dAf = torch.from_numpy(np.load(attn_dir / "dA_feat_1minus0.npy"))

    A0t = torch.from_numpy(np.load(attn_dir / "A_time_label0.npy"))
    A1t = torch.from_numpy(np.load(attn_dir / "A_time_label1.npy"))
    dAt = torch.from_numpy(np.load(attn_dir / "dA_time_1minus0.npy"))

    # Feature graph: 네트워크 플롯
    plot_topk_digraph(A0f, attn_dir / "feat_label0_topk.png", "FeatureGAT label0", topk=args.topk_feat)
    plot_topk_digraph(A1f, attn_dir / "feat_label1_topk.png", "FeatureGAT label1", topk=args.topk_feat)
    plot_topk_digraph(torch.clamp(dAf, min=0), attn_dir / "feat_diff_pos_topk.png", "FeatureGAT (label1-label0)+", topk=args.topk_feat)

    # Temporal graph: heatmap + (원하면) topk
    plot_heatmap(A0t, attn_dir / "time_label0_heat.png", "TemporalGAT label0")
    plot_heatmap(A1t, attn_dir / "time_label1_heat.png", "TemporalGAT label1")
    plot_heatmap(dAt,  attn_dir / "time_diff_heat.png", "TemporalGAT diff (1-0)")

    plot_topk_digraph(A0t, attn_dir / "time_label0_topk.png", "TemporalGAT label0 (topk)", topk=args.topk_time)
    plot_topk_digraph(A1t, attn_dir / "time_label1_topk.png", "TemporalGAT label1 (topk)", topk=args.topk_time)
    plot_topk_digraph(torch.clamp(dAt, min=0), attn_dir / "time_diff_pos_topk.png", "TemporalGAT (label1-label0)+ topk", topk=args.topk_time)

    print(f"[OK] plots saved under: {attn_dir}")


if __name__ == "__main__":
    main()

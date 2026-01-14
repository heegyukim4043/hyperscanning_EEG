import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split

from datasets.pd3_window_dataset import build_pd3_datasets
from models.hyperscan_mtad_gat_multilabel import HyperscanMTADGAT_MultiLabel

import time

def compute_pos_weight(loader):
    # pos_weight = neg/pos per label-dim
    y_all = []
    for _, y in loader:
        y_all.append(y)
    y = torch.cat(y_all, dim=0)  # [N,3]
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
    return pos_weight


@torch.no_grad()
def eval_model(model, loader, device, thr=0.5):
    model.eval()
    all_p, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        p = torch.sigmoid(logits)
        all_p.append(p.cpu())
        all_y.append(y.cpu())

    P = torch.cat(all_p, dim=0).numpy()
    Y = torch.cat(all_y, dim=0).numpy()

    pred = (P >= thr).astype(np.int32)
    Yb = (Y >= 0.5).astype(np.int32)

    # per-label precision/recall/f1
    stats = {}
    for k, name in enumerate(["i(1-2)", "j(1-3)", "k(2-3)"]):
        tp = ((pred[:, k] == 1) & (Yb[:, k] == 1)).sum()
        fp = ((pred[:, k] == 1) & (Yb[:, k] == 0)).sum()
        fn = ((pred[:, k] == 0) & (Yb[:, k] == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        stats[name] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

    # micro f1
    tp = ((pred == 1) & (Yb == 1)).sum()
    fp = ((pred == 1) & (Yb == 0)).sum()
    fn = ((pred == 0) & (Yb == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    micro_f1 = 2 * prec * rec / (prec + rec + 1e-9)
    stats["micro"] = {"precision": float(prec), "recall": float(rec), "f1": float(micro_f1)}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="datasets/PD3/processed")
    ap.add_argument("--lookback", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--save_dir", type=str, default="output_PD3_multilabel")
    args = ap.parse_args()

    device = torch.device("cuda:0" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"[Device] {device}")

    # build datasets
    train_sets, test_sets, bases = build_pd3_datasets(args.processed_dir, lookback=args.lookback, stride=args.stride)
    train_ds = ConcatDataset(train_sets)
    test_ds  = ConcatDataset(test_sets)

    n_total = len(train_ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.bs, shuffle=False)

    # models
    model = HyperscanMTADGAT_MultiLabel(
        n_features=57,
        window_size=args.lookback,
        gat_heads=2,
        dropout=0.25,
        alpha=0.2,
        use_gatv2=True,
        gru_hid_dim=150,
        gru_n_layers=1,
        fuse_mode="concat",
    ).to(device)

    # loss (pos_weight)
    tmp_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=False)
    pos_weight = compute_pos_weight(tmp_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_path = Path(args.save_dir) / "best_model.pt"

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / max(n, 1)

        val_stats = eval_model(model, val_loader, device)
        val_f1 = val_stats["micro"]["f1"]

        elapsed = time.perf_counter() - t0
        print(
            f"[Epoch {ep}] train_loss={train_loss:.6f} | val_micro_f1={val_f1:.4f} | "
            f"val_f1(i/j/k)=({val_stats['i(1-2)']['f1']:.3f}, "
            f"{val_stats['j(1-3)']['f1']:.3f}, "
            f"{val_stats['k(2-3)']['f1']:.3f}) "
            f"[{elapsed:.1f}s]"
        )

        if val_f1 > best_val:
            best_val = val_f1
            torch.save({"models": model.state_dict(), "args": vars(args)}, best_path)

    # test with best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["models"])
    test_stats = eval_model(model, test_loader, device)
    print("[TEST]", test_stats)

    # save test stats
    import json
    with open(Path(args.save_dir) / "test_stats.json", "w", encoding="utf-8") as f:
        json.dump(test_stats, f, indent=2)

    print(f"[DONE] saved: {best_path}")


if __name__ == "__main__":
    main()

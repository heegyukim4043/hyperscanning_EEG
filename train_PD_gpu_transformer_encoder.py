import os
import torch
import torch.nn as nn
import json
import numpy as np
from datetime import datetime

from arg_PD_transformer_encoder import get_parser   # ← 새 argparser 사용
from utils_PD import *
from gat_transformer_pre import MTAD_GAT_TransformerRecon  # ← 모델 임포트
from prediction import Predictor
from training_pre import Trainer

from mtad_gat_dgl_full import MTAD_GAT_DGL_Full

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception:
    nx = None


class WindowLabelDataset(Dataset):
    """
    x_raw: torch.Tensor [T,F] (CPU 텐서 권장)
    y_point: np.ndarray [T] or list[int] (0/1)
    window_size: W
    label_rule: "last" (윈도우 마지막 시점 라벨)
    """
    def __init__(self, x_raw, y_point, window_size: int):
        self.x = x_raw
        self.y = y_point
        self.W = window_size
        self.n = x_raw.shape[0] - window_size  # 윈도우 개수

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        xw = self.x[idx:idx + self.W]                 # [W,F]
        yw = int(self.y[idx + self.W])                # 마지막 시점 라벨 (y_test[lookback:]와 정렬 동일)
        return xw, yw


def _find_gat_modules(model):
    # class name 기반으로 찾아서 모델 내부 구조에 덜 의존하도록 구성
    feat_gat = None
    time_gat = None
    for m in model.modules():
        if m.__class__.__name__ == "DGLFeatureGAT":
            feat_gat = m
        elif m.__class__.__name__ == "DGLTemporalGAT":
            time_gat = m
    return feat_gat, time_gat


@torch.no_grad()
def compute_mean_attn_by_label(model, loader, device, which="feature"):
    model.eval()
    feat_gat, time_gat = _find_gat_modules(model)
    if which == "feature" and feat_gat is None:
        raise RuntimeError("DGLFeatureGAT를 models.modules()에서 찾지 못했습니다. (모델 내부 연결 확인 필요)")
    if which == "temporal" and time_gat is None:
        raise RuntimeError("DGLTemporalGAT를 models.modules()에서 찾지 못했습니다. (모델 내부 연결 확인 필요)")

    A0_sum, A1_sum = None, None
    n0, n1 = 0, 0

    for xw, yw in loader:
        xw = xw.to(device)
        yw = yw.to(device).long()

        if which == "feature":
            _, A = feat_gat(xw, return_attn=True)   # [B,F,F]
        else:
            _, A = time_gat(xw, return_attn=True)   # [B,W,W]

        idx0 = (yw == 0)
        idx1 = (yw == 1)

        if idx0.any():
            s0 = A[idx0].sum(dim=0)
            A0_sum = s0 if A0_sum is None else (A0_sum + s0)
            n0 += idx0.sum().item()

        if idx1.any():
            s1 = A[idx1].sum(dim=0)
            A1_sum = s1 if A1_sum is None else (A1_sum + s1)
            n1 += idx1.sum().item()

    A0 = A0_sum / max(n0, 1)
    A1 = A1_sum / max(n1, 1)
    dA = A1 - A0
    return A0, A1, dA, n0, n1


def plot_heatmap(A, out_png: Path, title: str):
    A = A.detach().float().cpu().numpy()
    plt.figure()
    plt.imshow(A, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_topk_digraph(A, out_png, title: str, topk: int = 250, seed: int = 0):
    """
    A: torch.Tensor [N,N] (attention adjacency)
    """
    if nx is None:
        print("[WARN] networkx가 없어 graph plot은 스킵합니다.")
        return

    import numpy as np
    from pathlib import Path

    out_png = Path(out_png)

    A = A.detach().float().cpu().numpy().copy()
    np.fill_diagonal(A, 0.0)

    N = A.shape[0]
    flat = A.ravel()
    k = min(topk, flat.size)

    # top-k edge 선택
    idx = np.argsort(flat)[::-1][:k]
    src, dst = np.unravel_index(idx, (N, N))

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for s, d in zip(src, dst):
        w = float(A[s, d])
        if w > 0:
            G.add_edge(int(s), int(d), weight=w)

    # --- 명시적으로 fig/ax 생성 (colorbar 오류 방지 핵심) ---
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.axis("off")

    # 노드만이라도 그림
    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_nodes(G, pos, node_size=25, ax=ax)

    # 엣지가 없으면 colorbar 없이 저장하고 종료
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

    # colorbar를 위한 mappable: set_array 필요(버전 호환성)
    vmax = max(1e-6, float(weights.max()))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])

    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)



if __name__ == "__main__":

    parser = get_parser()

    parser.add_argument("--analyze_attn", action="store_true",
                        help="학습 후 label(0/1)별 attention 그래프 평균/차이를 저장/시각화")
    parser.add_argument("--attn_topk_feat", type=int, default=250,
                        help="Feature graph에서 시각화할 top-k edges")
    parser.add_argument("--attn_topk_time", type=int, default=600,
                        help="Temporal graph에서 시각화할 top-k edges (W=150이면 600~2000 권장)")
    parser.add_argument("--attn_make_plots", action="store_true",
                        help="PNG 플롯 생성 여부")

    args = parser.parse_args()
    args_summary = str(args.__dict__)
    print(args_summary)

    # 반복할 dataset / group 리스트 지정
    datasets = ["PD2"]       # 필요시 ["PD", "PD2", "MSL", ...]
   # groups = ["2-1","3-1","4-1","5-1","6-1","7-1","8-1","9-1","10-1","11-1"]        # 원하는 그룹들
    groups = ["4-1", "5-1", "6-1", "7-1", "8-1", "9-1", "10-1", "11-1"]
    for dataset in datasets:
        for group in groups:



            id = datetime.now().strftime("%d%m%Y_%H%M%S")
            args.dataset = dataset
            args.group = group
            print('Check_group')
            print(group[0:2])
            print(group[3:])

            # 반드시 여기서 다시 만들고 출력
            args_summary = str(args.__dict__)
            print("\n==============================")
            print("[RUN]", "dataset=", args.dataset, "group=", args.group)
            print(args_summary)
            print("==============================\n")

            group_index = group.split("-")[0]
            index = group.split("-")[1]
            """
            if len(group) == 4:
                group_index = group[0:2]
                index = group[3:]

            elif len(group) == 3:
                group_index = group[0]
                index = group[2:]
            """
            group_index, index = group.split("-")  # "10-1" -> "10", "1"

            # 데이터 로드
            if dataset == "PD":
                output_path = f"output/PD/{group}"
                (x_train, _), (x_test, y_test) = get_data(
                    f"machine-{group_index}-{index}", normalize=args.normalize
                )
            elif dataset == "PD2":
                output_path = f"output/PD2/{group}"
                (x_train, _), (x_test, y_test) = get_data(
                    f"machine-{group_index}-{index}", normalize=args.normalize
                )
            else:
                raise Exception(f'Dataset "{dataset}" not available.')

            log_dir = f"{output_path}/logs"
            os.makedirs(log_dir, exist_ok=True)
            save_path = f"{output_path}/{id}"

            # Tensor 변환
            x_train = torch.from_numpy(x_train).float()
            x_test = torch.from_numpy(x_test).float()
            n_features = x_train.shape[1]

            device = torch.device(
                "cuda:0" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
            )
            x_train = x_train.to(device)
            x_test = x_test.to(device)

            print("CUDA available:", torch.cuda.is_available(), " | args.use_cuda:", args.use_cuda)
            print("Device Trainer will use:", device)

            # target dims
            target_dims = get_target_dims(dataset)
            if target_dims is None:
                out_dim = n_features
                print(f"Will forecast and reconstruct all {n_features} input features")
            elif isinstance(target_dims, int):
                print(f"Will forecast and reconstruct input feature: {target_dims}")
                out_dim = 1
            else:
                print(f"Will forecast and reconstruct input features: {target_dims}")
                out_dim = len(target_dims)

            # Dataset / Loader
            train_dataset = SlidingWindowDataset(x_train, args.lookback, target_dims)
            test_dataset = SlidingWindowDataset(x_test, args.lookback, target_dims)
            train_loader, val_loader, test_loader = create_data_loaders(
                train_dataset, args.bs, args.val_split, args.shuffle_dataset,
                test_dataset=test_dataset
            )

            # === 모델 생성 (Reconstruction만 Transformer) ===
            """
            models = MTAD_GAT_TransformerRecon(
                n_features,
                args.lookback,
                out_dim,
                gru_hid_dim=args.gru_hid_dim,
                forecast_hid_dim=args.fc_hid_dim,
                recon_d_model=args.recon_d_model,
                recon_nhead=args.recon_nhead,
                recon_num_layers=args.recon_num_layers,
                recon_dim_ff=args.recon_dim_ff,
                dropout=args.dropout,
            ).to(device)
            """

            model = MTAD_GAT_DGL_Full(
                n_features=n_features,
                window_size=args.lookback,
                out_dim=out_dim,
                kernel_size=args.kernel_size,
                use_gatv2=args.use_gatv2,
                gat_heads_feat=2,
                gat_heads_time=2,
                gru_n_layers=args.gru_n_layers,
                gru_hid_dim=args.gru_hid_dim,
                fc_n_layers=args.fc_n_layers,
                fc_hid_dim=args.fc_hid_dim,
                recon_d_model=args.recon_d_model,
                recon_nhead=args.recon_nhead,
                recon_num_layers=args.recon_num_layers,
                recon_dim_ff=args.recon_dim_ff,
                dropout=args.dropout,
                alpha=args.alpha,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
            forecast_criterion = nn.MSELoss()
            recon_criterion = nn.MSELoss()

            trainer = Trainer(
                model,
                optimizer,
                args.lookback,
                n_features,
                target_dims,
                args.epochs,
                args.bs,
                args.init_lr,
                forecast_criterion,
                recon_criterion,
                args.use_cuda,
                save_path,
                log_dir,
                args.print_every,
                args.log_tensorboard,
                args_summary,
            )

            trainer.fit(train_loader, val_loader)
            plot_losses(trainer.losses, save_path=save_path, plot=False)

            # Evaluate test set
            test_loss = trainer.evaluate(test_loader)
            print(f"[{dataset}-{group}] Test forecast loss: {test_loss[0]:.5f}")
            print(f"[{dataset}-{group}] Test reconstruction loss: {test_loss[1]:.5f}")
            print(f"[{dataset}-{group}] Test total loss: {test_loss[2]:.5f}")

            # Load best models for prediction
            trainer.load(f"{save_path}/models.pt")
            prediction_args = {
                "dataset": dataset,
                "target_dims": target_dims,
                "scale_scores": args.scale_scores,
                "level": 0.90,
                "q": 0.005,
                "dynamic_pot": args.dynamic_pot,
                "use_mov_av": args.use_mov_av,
                "gamma": args.gamma,
                "reg_level": 1,
                "save_path": save_path,
            }
            best_model = trainer.model
            predictor = Predictor(best_model, args.lookback, n_features, prediction_args)

            label = y_test[args.lookback:] if y_test is not None else None
            predictor.predict_anomalies(x_train, x_test, label)

            # predictor.predict_anomalies(x_train, x_test, label) 바로 아래에 추가

            if args.analyze_attn and (y_test is not None):
                attn_dir = Path(save_path) / "attn"
                attn_dir.mkdir(parents=True, exist_ok=True)

                # 분석용 데이터는 CPU에서 윈도우 생성 후 배치마다 GPU로 이동(안전/일반적)
                x_test_cpu = torch.from_numpy(x_test.detach().cpu().numpy()).float() if torch.is_tensor(
                    x_test) else torch.from_numpy(x_test).float()

                # y_test는 point label (0/1). 파일도 이런 형식으로 구성되어 있음.:contentReference[oaicite:6]{index=6}
                ds = WindowLabelDataset(x_test_cpu, y_test, window_size=args.lookback)
                dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)

                # Feature graph
                A0f, A1f, dAf, n0, n1 = compute_mean_attn_by_label(best_model, dl, device=device, which="feature")
                np.save(attn_dir / "A_feat_label0.npy", A0f.detach().cpu().numpy())
                np.save(attn_dir / "A_feat_label1.npy", A1f.detach().cpu().numpy())
                np.save(attn_dir / "dA_feat_1minus0.npy", dAf.detach().cpu().numpy())

                # Temporal graph
                A0t, A1t, dAt, n0t, n1t = compute_mean_attn_by_label(best_model, dl, device=device, which="temporal")
                np.save(attn_dir / "A_time_label0.npy", A0t.detach().cpu().numpy())
                np.save(attn_dir / "A_time_label1.npy", A1t.detach().cpu().numpy())
                np.save(attn_dir / "dA_time_1minus0.npy", dAt.detach().cpu().numpy())

                print(f"[ATTN] saved to: {attn_dir}")
                print(f"[ATTN] feature label0 n={n0}, label1 n={n1}")
                print(f"[ATTN] temporal label0 n={n0t}, label1 n={n1t}")

                if args.attn_make_plots:
                    # Feature는 network plot이 보기 좋음
                    plot_topk_digraph(A0f, attn_dir / "feat_label0_topk.png", f"FeatureGAT label0 (n={n0})",
                                      topk=args.attn_topk_feat)
                    plot_topk_digraph(A1f, attn_dir / "feat_label1_topk.png", f"FeatureGAT label1 (n={n1})",
                                      topk=args.attn_topk_feat)
                    plot_topk_digraph(dAf.clamp_min(0), attn_dir / "feat_diff_pos_topk.png",
                                      "FeatureGAT (label1-label0)+", topk=args.attn_topk_feat)

                    # Temporal은 heatmap이 안정적 (W=150이면 network plot이 너무 복잡해질 수 있음)
                    plot_heatmap(A0t, attn_dir / "time_label0_heat.png", "TemporalGAT label0")
                    plot_heatmap(A1t, attn_dir / "time_label1_heat.png", "TemporalGAT label1")
                    plot_heatmap(dAt, attn_dir / "time_diff_heat.png", "TemporalGAT diff (1-0)")

                    # temporal도 원하면 topk graph로 가능
                    plot_topk_digraph(A0t, attn_dir / "time_label0_topk.png", "TemporalGAT label0 (topk)",
                                      topk=args.attn_topk_time)
                    plot_topk_digraph(A1t, attn_dir / "time_label1_topk.png", "TemporalGAT label1 (topk)",
                                      topk=args.attn_topk_time)
                    plot_topk_digraph(dAt.clamp_min(0), attn_dir / "time_diff_pos_topk.png",
                                      "TemporalGAT (label1-label0)+ topk", topk=args.attn_topk_time)

            # Save config
            args_path = f"{save_path}/config.txt"
            with open(args_path, "w") as f:
                json.dump(args.__dict__, f, indent=2)

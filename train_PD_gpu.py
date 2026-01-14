import os
import torch
import torch.nn as nn
import json
import numpy as np
from datetime import datetime

from args_PD_transformer import get_parser
from utils_PD import *
from transformer import MTAD_GAT
from prediction import Predictor
from training_pre import Trainer

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    args_summary = str(args.__dict__)
    print(args_summary)

    # 반복할 dataset / group 리스트 지정
    datasets = ["PD"]       # 필요시 ["PD", "PD2", "MSL", ...]
    groups = ["2-1", "3-1", "4-1", "5-1", "6-1", "7-1", "8-1", "9-1", "10-1", "11-1"]        # 원하는 그룹들

    for dataset in datasets:
        for group in groups:

            id = datetime.now().strftime("%d%m%Y_%H%M%S")
            args.dataset = dataset
            args.group = group

            group_index = group[0]
            index = group[2:]

            if dataset == "PD":
                output_path = f"output/PD/{group}"
                (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=args.normalize)
            elif dataset == "PD2":
                output_path = f"output/PD2/{group}"
                (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=args.normalize)
            else:
                raise Exception(f'Dataset "{dataset}" not available.')

            log_dir = f"{output_path}/logs"
            os.makedirs(log_dir, exist_ok=True)
            save_path = f"{output_path}/{id}"

            # Tensor 변환
            x_train = torch.from_numpy(x_train).float()
            x_test = torch.from_numpy(x_test).float()
            n_features = x_train.shape[1]

            device = torch.device("cuda:0" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
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
                train_dataset, args.bs, args.val_split, args.shuffle_dataset, test_dataset=test_dataset
            )

            # Model
            model = MTAD_GAT(
                n_features,
                args.lookback,
                out_dim,
                kernel_size=args.kernel_size,
                use_gatv2=args.use_gatv2,
                feat_gat_embed_dim=args.feat_gat_embed_dim,
                time_gat_embed_dim=args.time_gat_embed_dim,
                gru_n_layers=args.gru_n_layers,
                gru_hid_dim=args.gru_hid_dim,
                forecast_n_layers=args.fc_n_layers,
                forecast_hid_dim=args.fc_hid_dim,
                recon_n_layers=args.recon_n_layers,
                recon_hid_dim=args.recon_hid_dim,
                dropout=args.dropout,
                alpha=args.alpha,
                use_transformer=args.use_transformer,
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                dim_feedforward=args.dim_feedforward,
                attn_dropout=args.attn_dropout,
                causal=bool(args.causal),
                use_last_step=bool(args.use_last_step),
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
            trainer.load(f"{save_path}/model.pt")
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

            # Save config
            args_path = f"{save_path}/config.txt"
            with open(args_path, "w") as f:
                json.dump(args.__dict__, f, indent=2)

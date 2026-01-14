import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_parser():
    parser = argparse.ArgumentParser()

    # --- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="PD2")
    parser.add_argument("--group", type=str, default="2-2",
                        help="Dataset group index, e.g. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=150)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)

    # --- Model params ---
    # Conv layer
    parser.add_argument("--kernel_size", type=int, default=7)

    # GAT
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)

    # GRU encoder
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=150)

    # Forecasting head
    parser.add_argument("--fc_n_layers", type=int, default=1)
    parser.add_argument("--fc_hid_dim", type=int, default=150)

    # === Transformer Reconstruction head ===
    parser.add_argument("--recon_d_model", type=int, default=128,
                        help="Transformer hidden dimension for reconstruction")
    parser.add_argument("--recon_nhead", type=int, default=4,
                        help="Number of attention heads in Transformer recon")
    parser.add_argument("--recon_num_layers", type=int, default=2,
                        help="Number of Transformer decoder layers")
    parser.add_argument("--recon_dim_ff", type=int, default=256,
                        help="Feedforward dimension in Transformer decoder")
    parser.add_argument("--recon_dropout", type=float, default=0.2,
                        help="Dropout for Transformer recon")

    # Other
    parser.add_argument("--alpha", type=float, default=0.2)

    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--bs", type=int, default=50)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")

    return parser

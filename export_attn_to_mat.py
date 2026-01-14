import argparse
from pathlib import Path
import numpy as np
from scipy.io import savemat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_dir", required=True, help=".../attn 폴더 경로")
    args = ap.parse_args()

    attn_dir = Path(args.attn_dir)

    mats = {
        "A_feat_label0": np.load(attn_dir / "A_feat_label0.npy"),
        "A_feat_label1": np.load(attn_dir / "A_feat_label1.npy"),
        "dA_feat_1minus0": np.load(attn_dir / "dA_feat_1minus0.npy"),
        "A_time_label0": np.load(attn_dir / "A_time_label0.npy"),
        "A_time_label1": np.load(attn_dir / "A_time_label1.npy"),
        "dA_time_1minus0": np.load(attn_dir / "dA_time_1minus0.npy"),
    }

    out_path = attn_dir / "attn_mats.mat"
    savemat(out_path, mats, do_compression=True)
    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()



## python export_attn_to_mat.py --attn_dir "output\PD2\2-1\22122025_183254\attn"

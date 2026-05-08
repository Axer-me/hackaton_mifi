from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT
TRAIN_CSV = BASELINE_DIR / "data" / "train_df_full_clean.csv"
OUT_DIR = BASELINE_DIR / "data" / "cv_splits"
CFG_DIR = BASELINE_DIR / "configs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
    y = df["result"].astype(int).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_paths = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), start=1):
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        tr_csv = OUT_DIR / f"train_fold{fold}.csv"
        va_csv = OUT_DIR / f"val_fold{fold}.csv"
        tr.to_csv(tr_csv, index=False, encoding="utf-8")
        va.to_csv(va_csv, index=False, encoding="utf-8")

        cfg = {
            "experiment_name": f"exp_convnext_base_clean_lr1e4_fold{fold}",
            "data_root": ".",
            "train_csv": f"data/cv_splits/train_fold{fold}.csv",
            "val_csv": f"data/cv_splits/val_fold{fold}.csv",
            "test_csv": "test_df.csv",
            "train_images_dir": "train_images_full_clean/train_images_full_clean",
            "val_images_dir": "train_images_full_clean/train_images_full_clean",
            "test_images_dir": "test_images/test_images",
            "num_classes": 20,
            "min_ratio": 0.0,
            "model_name": "convnext_base",
            "pretrained": True,
            "dropout": 0.3,
            "img_size": 224,
            "batch_size": 8,
            "num_epochs": 12,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "label_smoothing": 0.05,
            "use_weighted_loss": False,
            "preload": True,
            "warmup_epochs": 2,
            "early_stopping_patience": 4,
            "seed": 42,
            "output_dir": f"outputs/exp_convnext_base_clean_lr1e4_fold{fold}",
        }
        cfg_path = CFG_DIR / f"exp_convnext_base_clean_lr1e4_fold{fold}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        cfg_paths.append(cfg_path)

    for p in cfg_paths:
        cmd = [str(PYTHON), "src/train.py", "--config", str(p.relative_to(BASELINE_DIR))]
        print("Running:", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=str(BASELINE_DIR))
        if res.returncode != 0:
            raise SystemExit(res.returncode)

    print("3-fold run complete.", flush=True)


if __name__ == "__main__":
    main()

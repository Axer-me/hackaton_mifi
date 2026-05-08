"""
Build OOF-based meta-ensemble for 3-fold ConvNeXt models.

Pipeline:
1) Collect OOF probabilities from each fold model on its validation split.
2) Fit multinomial LogisticRegression as a meta-model (calibration/stacking).
3) Run test inference with all fold models (with TTA), average probabilities.
4) Apply meta-model and save final submission.

Example:
  python scripts/stacking_convnext_3fold.py --tta hflip scale_up scale_down
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import RoomDataset, get_transforms
from src.model import RoomClassifier


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(weights_path: Path, model_name: str, num_classes: int, device: torch.device) -> RoomClassifier:
    model = RoomClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.eval()
    return model


def _center_crop_tensor(x: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return x[:, :, top : top + crop_h, left : left + crop_w]


def _center_pad_tensor(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = max(out_h - h, 0)
    pad_w = max(out_w - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")


def apply_tta(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return images
    if mode == "hflip":
        return torch.flip(images, dims=[3])
    if mode == "scale_up":
        # Slight zoom-in: upscale then center crop back.
        up = F.interpolate(images, scale_factor=1.06, mode="bilinear", align_corners=False)
        return _center_crop_tensor(up, images.shape[2], images.shape[3])
    if mode == "scale_down":
        # Slight zoom-out: downscale then center pad back.
        down = F.interpolate(images, scale_factor=0.94, mode="bilinear", align_corners=False)
        return _center_pad_tensor(down, images.shape[2], images.shape[3])
    raise ValueError(f"Unknown TTA mode: {mode}")


@torch.no_grad()
def predict_proba(
    model: RoomClassifier,
    loader: DataLoader,
    device: torch.device,
    tta_modes: Iterable[str],
    is_test: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    tta_modes = list(tta_modes)
    if not tta_modes:
        raise ValueError("tta_modes must contain at least one mode")

    all_probs: list[np.ndarray] = []
    all_ids: list[str] = []
    all_labels: list[int] = []

    for batch in tqdm(loader, leave=False):
        if is_test:
            images, ids = batch
            labels = None
        else:
            images, labels = batch
            ids = None

        images = images.to(device)
        probs_acc = None
        for tta in tta_modes:
            aug = apply_tta(images, tta)
            logits = model(aug)
            probs = torch.softmax(logits, dim=1)
            probs_acc = probs if probs_acc is None else probs_acc + probs

        probs_acc = probs_acc / float(len(tta_modes))
        all_probs.append(probs_acc.cpu().numpy())

        if is_test:
            all_ids.extend([str(x) for x in ids])
        else:
            all_labels.extend(labels.numpy().tolist())

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0), dtype=np.float32)
    ids_np = np.array(all_ids, dtype=object) if is_test else np.array([], dtype=object)
    labels_np = np.array(all_labels, dtype=np.int64) if not is_test else None
    return probs_np, ids_np, labels_np


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def resolve_from_config(cfg_path: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    project_root = cfg_path.parent.parent
    return (project_root / p).resolve()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"TTA modes: {args.tta}", flush=True)

    cfg_paths = [Path(p).resolve() for p in args.fold_configs]
    cfgs = [load_yaml(p) for p in cfg_paths]
    num_classes = int(cfgs[0].get("num_classes", 20))
    model_name = str(cfgs[0].get("model_name", "convnext_base"))

    fold_models: list[RoomClassifier] = []
    for cfg_path, cfg in zip(cfg_paths, cfgs):
        exp_name = cfg["experiment_name"]
        weights_path = resolve_from_config(cfg_path, cfg["output_dir"]) / "best_model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights for {exp_name}: {weights_path}")
        fold_models.append(build_model(weights_path, model_name=model_name, num_classes=num_classes, device=device))

    # 1) OOF collection
    oof_ids: list[str] = []
    oof_y: list[int] = []
    oof_probs: list[np.ndarray] = []

    for model, cfg_path, cfg in zip(fold_models, cfg_paths, cfgs):
        val_csv = resolve_from_config(cfg_path, cfg["val_csv"])
        val_df = pd.read_csv(val_csv)
        val_ds = RoomDataset(
            val_df,
            images_dir=str(resolve_from_config(cfg_path, cfg["val_images_dir"])),
            transform=get_transforms(int(cfg["img_size"]), "val"),
            mode="val",
            preload=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size or int(cfg["batch_size"]) * 2,
            shuffle=False,
            num_workers=0,
        )
        probs, _, labels = predict_proba(
            model=model,
            loader=val_loader,
            device=device,
            tta_modes=args.tta,
            is_test=False,
        )
        oof_ids.extend(val_df["image_id_ext"].astype(str).tolist())
        oof_y.extend(labels.tolist() if labels is not None else [])
        oof_probs.append(probs)

    X_oof = np.concatenate(oof_probs, axis=0)
    y_oof = np.array(oof_y, dtype=np.int64)

    # 2) Fit meta-model
    meta = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial",
        C=args.meta_c,
        random_state=args.seed,
    )
    meta.fit(X_oof, y_oof)
    oof_pred = meta.predict(X_oof)
    oof_f1 = f1_score(y_oof, oof_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
    print(f"OOF macro F1 (meta model): {oof_f1:.4f}", flush=True)

    # 3) Test inference with fold-mean probabilities
    test_csv = resolve_from_config(cfg_paths[0], cfgs[0]["test_csv"])
    test_df = pd.read_csv(test_csv)
    test_ds = RoomDataset(
        test_df,
        images_dir=str(resolve_from_config(cfg_paths[0], cfgs[0]["test_images_dir"])),
        transform=get_transforms(int(cfgs[0]["img_size"]), "val"),
        mode="test",
        preload=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size or int(cfgs[0]["batch_size"]) * 2,
        shuffle=False,
        num_workers=0,
    )

    test_probs_all = []
    test_ids_ref = None
    for model in tqdm(fold_models, desc="Test folds"):
        probs, ids, _ = predict_proba(
            model=model,
            loader=test_loader,
            device=device,
            tta_modes=args.tta,
            is_test=True,
        )
        test_probs_all.append(probs)
        if test_ids_ref is None:
            test_ids_ref = ids

    test_probs_mean = np.mean(test_probs_all, axis=0)
    test_pred = meta.predict(test_probs_mean).astype(int)

    # 4) Save artifacts
    out_dir = ROOT / "outputs" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({"image_id_ext": test_ids_ref, "Predicted": test_pred})
    sub.to_csv(out_dir / "submission.csv", index=False)

    oof_df = pd.DataFrame(X_oof, columns=[f"p{i}" for i in range(num_classes)])
    oof_df.insert(0, "image_id_ext", oof_ids)
    oof_df["target"] = y_oof
    oof_df["pred_meta"] = oof_pred
    oof_df.to_csv(out_dir / "oof_meta.csv", index=False)

    with (out_dir / "meta_model.pkl").open("wb") as f:
        pickle.dump(meta, f)

    meta_info = {
        "fold_configs": [str(p) for p in cfg_paths],
        "tta_modes": list(args.tta),
        "meta_c": args.meta_c,
        "seed": args.seed,
        "oof_macro_f1_meta": float(oof_f1),
        "num_classes": num_classes,
        "model_name": model_name,
    }
    with (out_dir / "meta_info.json").open("w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_dir / 'submission.csv'}", flush=True)
    print(f"Saved: {out_dir / 'oof_meta.csv'}", flush=True)
    print(f"Saved: {out_dir / 'meta_model.pkl'}", flush=True)
    print(f"Saved: {out_dir / 'meta_info.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-configs",
        nargs="+",
        default=[
            "configs/exp_convnext_base_clean_lr1e4_fold1.yaml",
            "configs/exp_convnext_base_clean_lr1e4_fold2.yaml",
            "configs/exp_convnext_base_clean_lr1e4_fold3.yaml",
        ],
    )
    parser.add_argument(
        "--tta",
        nargs="+",
        default=["none", "hflip", "scale_up", "scale_down"],
        choices=["none", "hflip", "scale_up", "scale_down"],
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--meta-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="ensemble_convnext_3fold_stacking_tta")
    main(parser.parse_args())

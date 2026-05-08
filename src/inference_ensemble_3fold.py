"""
Inference for 3-fold ConvNeXt ensemble with optional flip-TTA.

Usage:
  python src/inference_ensemble_3fold.py --tta
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import RoomDataset, get_transforms
from src.model import RoomClassifier


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(weights_path: str, device: torch.device) -> RoomClassifier:
    model = RoomClassifier(
        model_name="convnext_base",
        num_classes=20,
        pretrained=False,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
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
        up = F.interpolate(images, scale_factor=1.06, mode="bilinear", align_corners=False)
        return _center_crop_tensor(up, images.shape[2], images.shape[3])
    if mode == "scale_down":
        down = F.interpolate(images, scale_factor=0.94, mode="bilinear", align_corners=False)
        return _center_pad_tensor(down, images.shape[2], images.shape[3])
    raise ValueError(f"Unknown TTA mode: {mode}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_cfg(args.config)
    test_df = pd.read_csv(cfg["test_csv"])
    test_ds = RoomDataset(
        test_df,
        cfg["test_images_dir"],
        get_transforms(cfg["img_size"], "val"),
        mode="test",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size or cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=0,
    )

    fold_weight_paths = [
        "outputs/exp_convnext_base_clean_lr1e4_fold1/best_model.pth",
        "outputs/exp_convnext_base_clean_lr1e4_fold2/best_model.pth",
        "outputs/exp_convnext_base_clean_lr1e4_fold3/best_model.pth",
    ]
    models = [build_model(p, device) for p in fold_weight_paths]

    image_ids = []
    preds = []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Inference 3-fold ensemble"):
            images = images.to(device)
            logits = None
            for m in models:
                out = None
                for tta_mode in args.tta:
                    aug = apply_tta(images, tta_mode)
                    cur = m(aug)
                    out = cur if out is None else out + cur
                out = out / len(args.tta)
                logits = out if logits is None else logits + out
            logits = logits / len(models)
            batch_preds = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)
            image_ids.extend(ids)

    is_tta = not (len(args.tta) == 1 and args.tta[0] == "none")
    out_dir = "outputs/ensemble_convnext_3fold_tta" if is_tta else "outputs/ensemble_convnext_3fold"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")
    sub = pd.DataFrame({"image_id_ext": image_ids, "Predicted": preds})
    sub.to_csv(out_path, index=False)

    print(f"Saved submission: {out_path}")
    print(sub["Predicted"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_convnext_base_clean_lr1e4_fold1.yaml")
    parser.add_argument(
        "--tta",
        nargs="+",
        default=["none"],
        choices=["none", "hflip", "scale_up", "scale_down"],
        help="TTA modes, e.g. --tta none hflip scale_up scale_down",
    )
    parser.add_argument("--batch_size", type=int, default=0)
    main(parser.parse_args())

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
                out = m(images)
                if args.tta:
                    out_flip = m(torch.flip(images, dims=[3]))
                    out = (out + out_flip) / 2.0
                logits = out if logits is None else logits + out
            logits = logits / len(models)
            batch_preds = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)
            image_ids.extend(ids)

    out_dir = "outputs/ensemble_convnext_3fold_tta" if args.tta else "outputs/ensemble_convnext_3fold"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")
    sub = pd.DataFrame({"image_id_ext": image_ids, "Predicted": preds})
    sub.to_csv(out_path, index=False)

    print(f"Saved submission: {out_path}")
    print(sub["Predicted"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_convnext_base_clean_lr1e4_fold1.yaml")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--batch_size", type=int, default=0)
    main(parser.parse_args())

"""
Inference script — generates submission.csv
Usage:  python src/inference.py --config configs/baseline.yaml --weights outputs/baseline_efficientnet_b0/best_model.pth
"""
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


def main(cfg: dict, weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_df = pd.read_csv(cfg["test_csv"])
    test_ds = RoomDataset(
        test_df, cfg["test_images_dir"],
        get_transforms(cfg["img_size"], "val"),
        mode="test",
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"] * 2,
                             shuffle=False, num_workers=0)

    model = RoomClassifier(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=False,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    image_ids, predictions = [], []
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Inference"):
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            predictions.extend(preds.tolist())
            image_ids.extend(ids)

    submission = pd.DataFrame({
        "image_id_ext": image_ids,
        "Predicted": predictions,
    })
    out_path = os.path.join(cfg["output_dir"], "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}  ({len(submission)} rows)")
    print(submission["Predicted"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.weights)

"""
Baseline training script.
Usage:  python src/train.py --config configs/baseline.yaml
"""
import argparse
import os
import random
import sys
import time
import yaml

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import RoomDataset, get_transforms, build_weighted_sampler
from src.model import RoomClassifier


CLASS_NAMES = [
    "кухня/столовая", "кухня-гостиная", "универсальная", "гостиная",
    "спальня", "кабинет", "детская", "ванная",
    "туалет", "санузел", "коридор/прихожая", "гардеробная",
    "балкон/лоджия", "вид из окна", "дом снаружи", "подъезд",
    "другое", "предметы интерьера", "без мебели", "без мебели(19)",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.time()

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} [{i+1}/{n_batches}] "
                  f"loss={total_loss/(i+1):.4f} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        labels=list(range(num_classes)), zero_division=0)
    return macro_f1, all_labels, all_preds


def main(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # --- Data ---
    train_df = pd.read_csv(cfg["train_csv"])
    val_df   = pd.read_csv(cfg["val_csv"])

    if cfg.get("min_ratio", 0.0) > 0:
        before = len(train_df)
        train_df = train_df[train_df["ratio"] >= cfg["min_ratio"]].reset_index(drop=True)
        print(f"Filtered by ratio>={cfg['min_ratio']}: {before} -> {len(train_df)}", flush=True)

    num_classes = cfg["num_classes"]
    img_size    = cfg["img_size"]

    preload = cfg.get("preload", False)
    train_ds = RoomDataset(train_df, cfg["train_images_dir"],
                           get_transforms(img_size, "train"), mode="train", preload=preload)
    val_ds   = RoomDataset(val_df,   cfg["val_images_dir"],
                           get_transforms(img_size, "val"),   mode="val", preload=preload)

    if cfg.get("use_weighted_loss", False):
        sampler = build_weighted_sampler(train_df)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  sampler=sampler, num_workers=0, pin_memory=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=0, pin_memory=False)

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2,
                            shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)

    # --- Model ---
    model = RoomClassifier(
        model_name=cfg["model_name"],
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {cfg['model_name']} | Params: {total_params:.1f}M", flush=True)

    # --- Loss & optimizer ---
    if cfg.get("use_weighted_loss", False):
        counts = train_df["result"].value_counts().sort_index()
        w = torch.tensor(
            [1.0 / counts.get(i, 1) for i in range(num_classes)],
            dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.1))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    total_epochs = int(cfg["num_epochs"])
    if warmup_epochs > 0 and warmup_epochs < total_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["num_epochs"]
        )

    # --- Training loop ---
    best_f1 = 0.0
    best_epoch = 0
    bad_epochs = 0
    early_stopping_patience = int(cfg.get("early_stopping_patience", 0))
    history = []
    best_model_path = os.path.join(cfg["output_dir"], "best_model.pth")

    print(f"\nStarting training for {cfg['num_epochs']} epochs...\n", flush=True)

    for epoch in range(1, cfg["num_epochs"] + 1):
        t_epoch = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        macro_f1, y_true, y_pred = evaluate(model, val_loader, device, num_classes)
        scheduler.step()

        epoch_time = time.time() - t_epoch
        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_macro_f1": macro_f1,
        })

        print(f"Epoch {epoch:2d}/{cfg['num_epochs']} | "
              f"loss={train_loss:.4f} | val_macro_f1={macro_f1:.4f} | "
              f"time={epoch_time:.0f}s", flush=True)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> New best model saved (F1={best_f1:.4f})", flush=True)
        else:
            bad_epochs += 1
            if early_stopping_patience > 0 and bad_epochs >= early_stopping_patience:
                print(
                    f"  >> Early stopping at epoch {epoch}: "
                    f"no improvement for {bad_epochs} epochs",
                    flush=True,
                )
                break

    # --- Final report ---
    print(f"\n{'='*60}", flush=True)
    print(
        f"Training complete. Best val Macro F1: {best_f1:.4f} (epoch {best_epoch})",
        flush=True,
    )

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, y_true, y_pred = evaluate(model, val_loader, device, num_classes)

    present_classes = sorted(set(y_true) | set(y_pred))
    present_names   = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
                       for c in present_classes]
    report = classification_report(y_true, y_pred,
                                   labels=present_classes,
                                   target_names=present_names,
                                   zero_division=0)
    print("\nClassification Report (best model):\n", flush=True)
    print(report, flush=True)

    # Save history and report
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(cfg["output_dir"], "history.csv"), index=False)

    with open(os.path.join(cfg["output_dir"], "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best Val Macro F1: {best_f1:.4f}\n\n")
        f.write(report)

    print(f"\nOutputs saved to: {cfg['output_dir']}", flush=True)
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)

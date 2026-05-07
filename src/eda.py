"""
EDA script — saves plots and stats to outputs/eda/
Usage: python src/eda.py
"""
import os
import sys
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "outputs/eda"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "кухня/столовая", "кухня-гостиная", "универсальная", "гостиная",
    "спальня", "кабинет", "детская", "ванная",
    "туалет", "санузел", "коридор/прихожая", "гардеробная",
    "балкон/лоджия", "вид из окна", "дом снаружи", "подъезд",
    "другое", "предметы интерьера", "без мебели", "без мебели(19)",
]

# ── Load data ──────────────────────────────────────────────────────────────────
train_df = pd.read_csv("data/train_df.csv")
val_df   = pd.read_csv("data/val_df.csv")
test_df  = pd.read_csv("data/test_df.csv")

print(f"Train: {train_df.shape}  Val: {val_df.shape}  Test: {test_df.shape}")

# ── 1. Class distribution ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

for ax, df, title in [(axes[0], train_df, "Train (4562)"), (axes[1], val_df, "Val (500)")]:
    counts = df["result"].value_counts().sort_index()
    labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in counts.index]
    colors = ["#e74c3c" if c < 100 else "#3498db" for c in counts.values]
    bars = ax.bar(range(len(counts)), counts.values, color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=7)
    red_patch  = mpatches.Patch(color="#e74c3c", label="< 100 samples (critical)")
    blue_patch = mpatches.Patch(color="#3498db", label="≥ 100 samples")
    ax.legend(handles=[red_patch, blue_patch])

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: 01_class_distribution.png")

# ── 2. Ratio distribution ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train_df["ratio"], bins=20, color="#2ecc71", edgecolor="white")
ax.axvline(0.6, color="red", linestyle="--", label="ratio=0.6 (filter threshold)")
ax.axvline(0.7, color="orange", linestyle="--", label="ratio=0.7")
noisy = (train_df["ratio"] < 0.6).sum()
ax.set_title(f"Labeling confidence (ratio)\nSamples with ratio<0.6: {noisy} ({noisy/len(train_df)*100:.1f}%)")
ax.set_xlabel("ratio (fraction of labelers who agreed)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_ratio_distribution.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: 02_ratio_distribution.png")

# ── 3. Sample images per class ────────────────────────────────────────────────
IMG_DIR = "data/train_images/train_images"
n_cols = 5
present_classes = sorted(train_df["result"].unique())

for class_id in present_classes:
    cls_df = train_df[train_df["result"] == class_id].sample(
        min(5, len(train_df[train_df["result"] == class_id])), random_state=42
    )
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 3))
    cls_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
    fig.suptitle(f"Class {class_id}: {cls_name} ({len(train_df[train_df['result']==class_id])} samples)", fontsize=11)

    for j, ax in enumerate(axes):
        ax.axis("off")
        if j < len(cls_df):
            img_path = os.path.join(IMG_DIR, f"{cls_df.iloc[j]['image_id_ext']}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                ax.imshow(np.array(img))
                ax.set_title(f"ratio={cls_df.iloc[j]['ratio']:.2f}", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_samples_class{class_id:02d}.png", dpi=100, bbox_inches="tight")
    plt.close()

print(f"Saved: sample images for {len(present_classes)} classes")

# ── 4. Statistics summary ─────────────────────────────────────────────────────
stats = {
    "total_train": len(train_df),
    "total_val": len(val_df),
    "total_test": len(test_df),
    "num_classes_train": train_df["result"].nunique(),
    "num_classes_val": val_df["result"].nunique(),
    "ratio_below_0.6": int((train_df["ratio"] < 0.6).sum()),
    "ratio_below_0.7": int((train_df["ratio"] < 0.7).sum()),
    "min_class_count": int(train_df["result"].value_counts().min()),
    "max_class_count": int(train_df["result"].value_counts().max()),
    "rarest_class": int(train_df["result"].value_counts().idxmin()),
}

print("\n=== EDA Summary ===")
for k, v in stats.items():
    print(f"  {k}: {v}")

per_class = train_df["result"].value_counts().sort_index()
per_class.index = [f"{i}:{CLASS_NAMES[i]}" if i < len(CLASS_NAMES) else str(i)
                   for i in per_class.index]
per_class.to_csv(f"{OUT_DIR}/class_counts.csv")
print(f"\nAll EDA outputs saved to: {OUT_DIR}/")

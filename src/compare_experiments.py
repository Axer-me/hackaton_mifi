"""
Compare experiment results from history.csv files.
Usage: python src/compare_experiments.py
"""
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = "outputs"
results = []

for hist_path in glob.glob(f"{out_dir}/*/history.csv"):
    exp_name = os.path.basename(os.path.dirname(hist_path))
    df = pd.read_csv(hist_path)
    best_f1 = df["val_macro_f1"].max()
    best_epoch = df.loc[df["val_macro_f1"].idxmax(), "epoch"]
    results.append({
        "experiment": exp_name,
        "best_val_macro_f1": round(best_f1, 4),
        "best_epoch": int(best_epoch),
        "final_epoch": int(df["epoch"].max()),
    })

if not results:
    print("No experiment results found yet.")
else:
    summary = pd.DataFrame(results).sort_values("best_val_macro_f1", ascending=False)
    print("\n=== Experiment Comparison ===")
    print(summary.to_string(index=False))
    summary.to_csv(f"{out_dir}/experiment_summary.csv", index=False)

    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for hist_path in glob.glob(f"{out_dir}/*/history.csv"):
        exp_name = os.path.basename(os.path.dirname(hist_path))
        df = pd.read_csv(hist_path)
        axes[0].plot(df["epoch"], df["train_loss"], label=exp_name, marker="o", markersize=3)
        axes[1].plot(df["epoch"], df["val_macro_f1"], label=exp_name, marker="o", markersize=3)

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Val Macro F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].axhline(0.60, color="green", linestyle="--", alpha=0.6, label="target=0.60")
    axes[1].axhline(0.50, color="orange", linestyle="--", alpha=0.6, label="target=0.50")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/learning_curves.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_dir}/learning_curves.png")
    print(f"Saved: {out_dir}/experiment_summary.csv")

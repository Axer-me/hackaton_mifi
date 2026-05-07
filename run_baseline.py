"""Launcher for baseline training — run from project root."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
os.environ["PYTHONIOENCODING"] = "utf-8"

import yaml
from src.train import main

with open("configs/baseline.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=" * 60)
print(f"Experiment: {cfg['experiment_name']}")
print(f"Model:      {cfg['model_name']}")
print(f"Epochs:     {cfg['num_epochs']}")
print(f"Output:     {cfg['output_dir']}")
print("=" * 60)

best_f1 = main(cfg)
print(f"\nFINAL BEST VAL MACRO F1: {best_f1:.4f}")

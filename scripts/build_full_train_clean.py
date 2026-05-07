from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FULL_CSV = ROOT / "baseline" / "data" / "train_df_full.csv"
FULL_IMAGES_DIR = ROOT / "train_images_full" / "train_images_full"

OUT_CSV = ROOT / "baseline" / "data" / "train_df_full_clean.csv"
OUT_IMAGES_DIR = ROOT / "train_images_full_clean" / "train_images_full_clean"
LOG_DIR = ROOT / "baseline" / "outputs" / "full_train_clean_build"

# Sources with lower similarity to Avito photos (catalog/e-com style).
BLOCKED_DOMAIN_PARTS = [
    "ozone.",
    "wbbasket.",
    "wildberries.",
    "joom",
    "ebay.",
    "laredoute.",
    "mhid.",
    "usmall.",
    "ydo",
    "avatars.mds.yandex",
]

# Limit how many team samples per class are kept relative to base class size.
TEAM_CLASS_GROWTH_CAP = 0.40


def extract_domain(url: str) -> str:
    try:
        return (urlparse(str(url)).netloc or "").lower()
    except Exception:
        return ""


def is_domain_allowed(domain: str) -> bool:
    if not domain:
        return False
    return not any(part in domain for part in BLOCKED_DOMAIN_PARTS)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FULL_CSV)
    df["source"] = df.get("source", "base_train").fillna("base_train").astype(str)
    df["image_id_ext"] = df["image_id_ext"].astype(str)
    df["result"] = pd.to_numeric(df["result"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["result", "image", "image_id_ext"]).copy()
    df["result"] = df["result"].astype(int)

    base_df = df[df["source"] == "base_train"].copy()
    team_df = df[df["source"] != "base_train"].copy()

    # 1) Domain filter for team data.
    team_df["domain"] = team_df["image"].astype(str).apply(extract_domain)
    team_df["domain_allowed"] = team_df["domain"].apply(is_domain_allowed)
    team_df_filtered = team_df[team_df["domain_allowed"]].copy()

    # 2) Cap number of team samples per class.
    base_counts = base_df["result"].value_counts().to_dict()
    kept_team_parts = []
    dropped_team_parts = []
    for cls, cls_df in team_df_filtered.groupby("result"):
        cap = max(1, int(base_counts.get(cls, 0) * TEAM_CLASS_GROWTH_CAP))
        keep = cls_df.head(cap).copy()
        drop = cls_df.iloc[cap:].copy()
        kept_team_parts.append(keep)
        if len(drop) > 0:
            dropped_team_parts.append(drop)

    team_kept = pd.concat(kept_team_parts, ignore_index=True) if kept_team_parts else team_df_filtered.iloc[0:0].copy()
    team_dropped_cap = pd.concat(dropped_team_parts, ignore_index=True) if dropped_team_parts else team_df_filtered.iloc[0:0].copy()

    clean_df = pd.concat([base_df, team_kept], ignore_index=True)
    clean_df = clean_df.drop_duplicates(subset=["image_id_ext"], keep="first").copy()
    clean_df = clean_df.drop_duplicates(subset=["image"], keep="first").copy()

    # Save artifacts
    clean_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    if len(team_df) > 0:
        team_df[~team_df["domain_allowed"]].to_csv(LOG_DIR / "team_dropped_by_domain.csv", index=False, encoding="utf-8")
    if len(team_dropped_cap) > 0:
        team_dropped_cap.to_csv(LOG_DIR / "team_dropped_by_cap.csv", index=False, encoding="utf-8")

    # Copy selected images from full folder.
    copied = 0
    missing = []
    for img_id in clean_df["image_id_ext"].astype(str):
        src = FULL_IMAGES_DIR / f"{img_id}.jpg"
        dst = OUT_IMAGES_DIR / f"{img_id}.jpg"
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        else:
            missing.append({"image_id_ext": img_id})

    if missing:
        pd.DataFrame(missing).to_csv(LOG_DIR / "missing_images.csv", index=False, encoding="utf-8")

    print("Clean train build complete.", flush=True)
    print(f"Saved CSV: {OUT_CSV}", flush=True)
    print(f"Saved images dir: {OUT_IMAGES_DIR}", flush=True)
    print(f"Rows: {len(clean_df)}", flush=True)
    print(f"Copied images: {copied}", flush=True)
    print("Class distribution:", flush=True)
    print(clean_df["result"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()

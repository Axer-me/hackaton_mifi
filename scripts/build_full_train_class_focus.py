from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC_CSV = ROOT / "baseline" / "data" / "train_df_full.csv"
SRC_IMG_DIR = ROOT / "train_images_full" / "train_images_full"

OUT_CSV = ROOT / "baseline" / "data" / "train_df_full_class_focus.csv"
OUT_IMG_DIR = ROOT / "train_images_full_class_focus" / "train_images_full_class_focus"
LOG_DIR = ROOT / "baseline" / "outputs" / "full_train_class_focus_build"

# Stricter rules for known difficult classes.
FOCUS_CLASSES = {2, 5, 6, 17}
DEFAULT_CAP = 0.40
FOCUS_CAP = 0.20

BLOCKED_DOMAIN_PARTS = [
    "ozone.",
    "wbbasket.",
    "wildberries.",
    "joom",
    "ebay.",
    "laredoute.",
    "ydo",
    "avatars.mds.yandex",
]


def domain(url: str) -> str:
    try:
        return (urlparse(str(url)).netloc or "").lower()
    except Exception:
        return ""


def allowed(d: str) -> bool:
    if not d:
        return False
    return not any(x in d for x in BLOCKED_DOMAIN_PARTS)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC_CSV)
    if "source" not in df.columns:
        raise RuntimeError("train_df_full.csv must contain 'source' column.")
    df["result"] = pd.to_numeric(df["result"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["result", "image_id_ext", "image"]).copy()
    df["result"] = df["result"].astype(int)
    df["image_id_ext"] = df["image_id_ext"].astype(str)

    base = df[df["source"] == "base_train"].copy()
    team = df[df["source"] != "base_train"].copy()
    team["domain"] = team["image"].astype(str).apply(domain)
    team = team[team["domain"].apply(allowed)].copy()

    base_counts = base["result"].value_counts().to_dict()
    keep_parts = []
    drop_parts = []
    for cls, grp in team.groupby("result"):
        cap_ratio = FOCUS_CAP if cls in FOCUS_CLASSES else DEFAULT_CAP
        cap = max(1, int(base_counts.get(cls, 0) * cap_ratio))
        keep_parts.append(grp.head(cap))
        if len(grp) > cap:
            drop_parts.append(grp.iloc[cap:])

    team_keep = pd.concat(keep_parts, ignore_index=True) if keep_parts else team.iloc[0:0].copy()
    team_drop = pd.concat(drop_parts, ignore_index=True) if drop_parts else team.iloc[0:0].copy()

    out = pd.concat([base, team_keep], ignore_index=True)
    out = out.drop_duplicates(subset=["image_id_ext"], keep="first")
    out = out.drop_duplicates(subset=["image"], keep="first")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    if len(team_drop) > 0:
        team_drop.to_csv(LOG_DIR / "dropped_by_class_cap.csv", index=False, encoding="utf-8")

    copied = 0
    missing = []
    for img_id in out["image_id_ext"].astype(str):
        src = SRC_IMG_DIR / f"{img_id}.jpg"
        dst = OUT_IMG_DIR / f"{img_id}.jpg"
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        else:
            missing.append({"image_id_ext": img_id})

    if missing:
        pd.DataFrame(missing).to_csv(LOG_DIR / "missing_images.csv", index=False, encoding="utf-8")

    print("Class-focus dataset built.")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved images: {OUT_IMG_DIR}")
    print(f"Rows: {len(out)}")
    print("Class distribution:")
    print(out["result"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

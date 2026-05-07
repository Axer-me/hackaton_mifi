from __future__ import annotations

import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = ROOT / "Собрано нами"
BASE_TRAIN_CSV = ROOT / "train_df.csv"
BASE_IMAGES_DIR = ROOT / "train_images" / "train_images"

OUT_CSV = ROOT / "baseline" / "data" / "train_df_full.csv"
OUT_IMAGES_DIR = ROOT / "train_images_full" / "train_images_full"
LOG_DIR = ROOT / "baseline" / "outputs" / "full_train_build"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_auto(path: Path, sep: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=sep)


def normalize_team_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rename_map = {
        "item_id_ext": "item_id",
        "img_url": "image",
        "url": "image",
        "target": "result",
        "class_id": "result",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    required = ["item_id", "image", "image_id_ext", "result", "label", "ratio"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    df["image"] = df["image"].astype(str).str.strip()
    df = df[df["image"].str.startswith("http", na=False)].copy()
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.dropna(subset=["result"]).copy()
    df["result"] = df["result"].astype(int)
    df = df[(df["result"] >= 0) & (df["result"] <= 19)].copy()

    # Team CSVs usually do not have image_id_ext.
    missing_id = df["image_id_ext"].isna() | (df["image_id_ext"].astype(str).str.strip() == "")
    df.loc[missing_id, "image_id_ext"] = df.loc[missing_id, "image"].apply(
        lambda u: "ext_" + hashlib.md5(str(u).encode("utf-8")).hexdigest()
    )
    df["image_id_ext"] = df["image_id_ext"].astype(str).str.strip()

    df["item_id"] = df["item_id"].fillna(-1)
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce").fillna(1.0)
    df["source"] = source
    return df[required + ["source"]]


def copy_base_images(df: pd.DataFrame) -> int:
    copied = 0
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for img_id in df["image_id_ext"].astype(str):
        src = BASE_IMAGES_DIR / f"{img_id}.jpg"
        dst = OUT_IMAGES_DIR / f"{img_id}.jpg"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def download_one(url: str, img_id: str, timeout: int = 15) -> tuple[str, bool, str]:
    dst = OUT_IMAGES_DIR / f"{img_id}.jpg"
    if dst.exists():
        return img_id, True, "exists"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; avito-hackaton-bot/1.0)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        if r.status_code != 200 or not r.content:
            return img_id, False, f"http_{r.status_code}"
        dst.write_bytes(r.content)
        return img_id, True, "downloaded"
    except Exception as e:
        return img_id, False, f"error:{type(e).__name__}"


def download_team_images(df: pd.DataFrame, workers: int = 16) -> tuple[set[str], list[dict]]:
    successes: set[str] = set()
    fails: list[dict] = []

    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for row in df.itertuples(index=False):
            tasks.append(ex.submit(download_one, row.image, str(row.image_id_ext)))

        total = len(tasks)
        for i, fut in enumerate(as_completed(tasks), start=1):
            img_id, ok, reason = fut.result()
            if ok:
                successes.add(img_id)
            else:
                fails.append({"image_id_ext": img_id, "reason": reason})

            if i % 50 == 0 or i == total:
                print(f"  Download progress: {i}/{total} | ok={len(successes)} fail={len(fails)}", flush=True)

    return successes, fails


def main() -> None:
    print("Reading base train...", flush=True)
    base_df = pd.read_csv(BASE_TRAIN_CSV)
    base_df["source"] = "base_train"
    base_df = base_df[["item_id", "image", "image_id_ext", "result", "label", "ratio", "source"]].copy()
    base_df["image_id_ext"] = base_df["image_id_ext"].astype(str)

    print("Reading team CSVs...", flush=True)
    team_frames = []
    for csv_path in sorted(TEAM_DIR.glob("*.csv")):
        df_raw = read_csv_auto(csv_path, sep=";")
        df_team = normalize_team_df(df_raw, source=f"team::{csv_path.name}")
        team_frames.append(df_team)
        print(f"  {csv_path.name}: {len(df_team)} valid labeled rows", flush=True)

    team_df = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame(columns=base_df.columns)

    # De-duplicate team by URL and generated ID.
    team_df = team_df.drop_duplicates(subset=["image"]).drop_duplicates(subset=["image_id_ext"]).copy()

    print(f"Base rows: {len(base_df)} | Team rows (valid): {len(team_df)}", flush=True)

    print("Preparing full image folder...", flush=True)
    copied = copy_base_images(base_df)
    print(f"  Copied base images: {copied}", flush=True)

    print("Downloading team images...", flush=True)
    team_success_ids, team_fails = download_team_images(team_df)

    if team_fails:
        pd.DataFrame(team_fails).to_csv(LOG_DIR / "team_download_failures.csv", index=False, encoding="utf-8")
        print(f"  Saved failures: {LOG_DIR / 'team_download_failures.csv'}", flush=True)

    team_df_ok = team_df[team_df["image_id_ext"].astype(str).isin(team_success_ids)].copy()
    full_df = pd.concat([base_df, team_df_ok], ignore_index=True)

    # Final deduplication: base keeps priority over team.
    full_df = full_df.drop_duplicates(subset=["image_id_ext"], keep="first").copy()
    full_df = full_df.drop_duplicates(subset=["image"], keep="first").copy()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\nBuild complete.", flush=True)
    print(f"Saved CSV: {OUT_CSV}", flush=True)
    print(f"Saved images dir: {OUT_IMAGES_DIR}", flush=True)
    print(f"Rows in train_df_full: {len(full_df)}", flush=True)
    print("Class distribution:", flush=True)
    print(full_df["result"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()

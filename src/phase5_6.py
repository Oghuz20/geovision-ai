"""Phases 5 + 6 -- land-use statistics and change detection.

Run from the repo root:  python phase5_6.py
Reads : data/2021_landcover.npy, data/2026_landcover.npy
Writes: data/transition_matrix.csv, data/nonbuiltup_to_builtup_mask.npy
"""
from pathlib import Path

import numpy as np
import pandas as pd


class CFG:
    DATA_DIR = Path("data")

    DATE1_LABEL, DATE2_LABEL = "2021", "2026"
    DATE1_LC_PATH = DATA_DIR / "2021_landcover.npy"
    DATE2_LC_PATH = DATA_DIR / "2026_landcover.npy"
    TRANSITION_CSV_PATH = DATA_DIR / "transition_matrix.csv"
    EXPANSION_MASK_PATH = DATA_DIR / "nonbuiltup_to_builtup_mask.npy"

    NUM_CLASSES = 7
    IGNORE_INDEX = 0
    PIXELS_PER_KM2 = 10_000          # 10 m pixels

    CLASS_NAMES = ["Ignore", "Tree cover", "Low vegetation", "Cropland",
                   "Built-up", "Bare/sparse", "Water/wetland"]
    GREEN_IDS = (1, 2)
    BUILTUP_ID = 4


# ---------------------------------------------------------------- Phase 5
def compute_area_stats(landcover):
    counts = np.bincount(landcover.flatten(), minlength=CFG.NUM_CLASSES)
    valid = counts[1:].sum()
    stats = {}
    for i in range(1, CFG.NUM_CLASSES):
        stats[CFG.CLASS_NAMES[i]] = {
            "km2": counts[i] / CFG.PIXELS_PER_KM2,
            "pct": (counts[i] / valid * 100) if valid > 0 else 0.0,
        }
    green_px = sum(counts[c] for c in CFG.GREEN_IDS)
    stats["Green space"] = {
        "km2": green_px / CFG.PIXELS_PER_KM2,
        "pct": (green_px / valid * 100) if valid > 0 else 0.0,
    }
    return stats


# ---------------------------------------------------------------- Phase 6
def build_transition_matrix(date1, date2):
    valid = (date1 != CFG.IGNORE_INDEX) & (date2 != CFG.IGNORE_INDEX)
    d1, d2 = date1[valid], date2[valid]
    matrix = np.zeros((CFG.NUM_CLASSES, CFG.NUM_CLASSES), dtype=np.int64)
    for i in range(CFG.NUM_CLASSES):
        for j in range(CFG.NUM_CLASSES):
            matrix[i, j] = np.sum((d1 == i) & (d2 == j))
    return matrix


def compute_nonbuiltup_to_builtup(date1, date2):
    was_builtup = date1 == CFG.BUILTUP_ID
    now_builtup = date2 == CFG.BUILTUP_ID
    was_ignore = (date1 == CFG.IGNORE_INDEX) | (date2 == CFG.IGNORE_INDEX)
    return now_builtup & ~was_builtup & ~was_ignore


def main():
    date1_landcover = np.load(CFG.DATE1_LC_PATH)
    date2_landcover = np.load(CFG.DATE2_LC_PATH)

    # ---- Phase 5: statistics
    stats1 = compute_area_stats(date1_landcover)
    stats2 = compute_area_stats(date2_landcover)
    for label, stats in [(CFG.DATE1_LABEL, stats1), (CFG.DATE2_LABEL, stats2)]:
        print(f"\n=== {label} ===")
        for name, v in stats.items():
            print(f"  {name:16s}: {v['km2']:7.3f} km^2  ({v['pct']:5.2f}%)")

    # ---- Phase 6: change detection
    matrix = build_transition_matrix(date1_landcover, date2_landcover)
    matrix_df = pd.DataFrame(matrix, index=CFG.CLASS_NAMES, columns=CFG.CLASS_NAMES)
    matrix_df.to_csv(CFG.TRANSITION_CSV_PATH)

    expansion_mask = compute_nonbuiltup_to_builtup(date1_landcover, date2_landcover)
    np.save(CFG.EXPANSION_MASK_PATH, expansion_mask)

    total_expansion_km2 = expansion_mask.sum() / CFG.PIXELS_PER_KM2
    print(f"\nTotal non-built-up -> built-up (gross gain): {total_expansion_km2:.3f} km^2")
    print("By source class:")
    for src in CFG.CLASS_NAMES:
        if src in ("Ignore", "Built-up"):
            continue
        px = matrix_df.loc[src, "Built-up"]
        if px > 0:
            print(f"  {src:16s} -> Built-up: {px / CFG.PIXELS_PER_KM2:7.3f} km^2 "
                  f"({px / expansion_mask.sum() * 100:4.1f}% of gross gain)")

    builtup1 = matrix_df.loc["Built-up"].sum() / CFG.PIXELS_PER_KM2
    builtup2 = matrix_df["Built-up"].sum() / CFG.PIXELS_PER_KM2
    reverse_km2 = matrix_df.loc["Built-up"].drop("Built-up").sum() / CFG.PIXELS_PER_KM2
    print(f"\nBuilt-up {CFG.DATE1_LABEL}: {builtup1:.3f} km^2 | {CFG.DATE2_LABEL}: {builtup2:.3f} km^2 "
          f"| Net growth: {builtup2 - builtup1:+.3f} km^2 ({(builtup2 - builtup1) / builtup1 * 100:+.1f}%)")
    print(f"Gross gain {total_expansion_km2:.3f} km^2 | reverse flow (built-up -> other) "
          f"{reverse_km2:.3f} km^2  <- a large reverse flow indicates classification noise")

    print("\nSaved:", CFG.TRANSITION_CSV_PATH, "and", CFG.EXPANSION_MASK_PATH)


if __name__ == "__main__":
    main()
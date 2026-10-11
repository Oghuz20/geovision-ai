"""Phase 8 -- rule-based planning indicators.

Run from the repo root:  python phase8_planning_indicators.py
Reads : data/2021_landcover.npy, data/2026_landcover.npy,
        data/nonbuiltup_to_builtup_mask.npy, data/hotspot_grid.npy

Thresholds are illustrative, not official planning standards, and every
indicator inherits the classification errors of the underlying maps.
"""
from pathlib import Path

import numpy as np


class CFG:
    DATA_DIR = Path("data")
    DATE1_LC_PATH = DATA_DIR / "2021_landcover.npy"
    DATE2_LC_PATH = DATA_DIR / "2026_landcover.npy"
    EXPANSION_MASK_PATH = DATA_DIR / "nonbuiltup_to_builtup_mask.npy"
    HOTSPOT_GRID_PATH = DATA_DIR / "hotspot_grid.npy"

    NUM_CLASSES = 7
    PIXELS_PER_KM2 = 10_000
    CLASS_NAMES = ["Ignore", "Tree cover", "Low vegetation", "Cropland",
                   "Built-up", "Bare/sparse", "Water/wetland"]
    GREEN_IDS = (1, 2)
    BUILTUP_ID = 4
    CROPLAND_ID, LOWVEG_ID = 3, 2


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


def compute_planning_indicators(stats1, stats2, expansion_mask, hotspot_grid):
    indicators = []

    green_pct = stats2["Green space"]["pct"]
    status = "OK" if green_pct >= 15.0 else "FLAG"
    indicators.append(("Vegetated cover (tree + low vegetation)", status, f"{green_pct:.1f}%"))

    crop1, crop2 = stats1["Cropland"]["km2"], stats2["Cropland"]["km2"]
    loss_pct = (crop1 - crop2) / crop1 * 100 if crop1 > 0 else 0.0
    status = "FLAG" if loss_pct >= 25.0 else "WATCH"
    indicators.append(("Agricultural land loss", status, f"{loss_pct:.0f}%"))

    builtup2 = stats2["Built-up"]["pct"]
    status = "FLAG" if builtup2 >= 60 else ("WATCH" if builtup2 >= 55 else "OK")
    indicators.append(("Built-up saturation", status, f"{builtup2:.1f}%"))

    total_km2 = expansion_mask.sum() / CFG.PIXELS_PER_KM2
    top10_km2 = np.sort(hotspot_grid.flatten())[::-1][:10].sum() / CFG.PIXELS_PER_KM2
    diffuse_pct = (total_km2 - top10_km2) / total_km2 * 100 if total_km2 > 0 else 0
    status = "FLAG" if diffuse_pct >= 70 else "OK"
    indicators.append(("Development pattern", status, f"{diffuse_pct:.0f}% diffuse"))

    return indicators


def main():
    lc1 = np.load(CFG.DATE1_LC_PATH)
    lc2 = np.load(CFG.DATE2_LC_PATH)
    expansion_mask = np.load(CFG.EXPANSION_MASK_PATH)
    hotspot_grid = np.load(CFG.HOTSPOT_GRID_PATH)

    stats1, stats2 = compute_area_stats(lc1), compute_area_stats(lc2)
    for name, status, value in compute_planning_indicators(stats1, stats2, expansion_mask, hotspot_grid):
        print(f"[{status:5s}] {name}: {value}")

    # Context for the cropland indicator: where did the mapped cropland go?
    crop_to_built = ((lc1 == CFG.CROPLAND_ID) & (lc2 == CFG.BUILTUP_ID)).sum() / CFG.PIXELS_PER_KM2
    crop_to_lowveg = ((lc1 == CFG.CROPLAND_ID) & (lc2 == CFG.LOWVEG_ID)).sum() / CFG.PIXELS_PER_KM2
    change = stats2["Cropland"]["km2"] - stats1["Cropland"]["km2"]
    print(f"\nMapped cropland change: {change:+.2f} km^2 "
          f"({crop_to_built:.2f} km^2 -> built-up, {crop_to_lowveg:.2f} km^2 -> low vegetation, "
          "a known confusion pair).")
    if change < 0:
        print("Cropland test IoU is only 0.34: treat this as a monitoring signal, not a confirmed loss.")


if __name__ == "__main__":
    main()
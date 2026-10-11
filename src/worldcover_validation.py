"""Consistency checks against reference data (neither is ground truth).

Run from the repo root:  python worldcover_validation.py
Reads : data/2021_landcover.npy, data/2026_landcover.npy,
        data/2021_RGBNIR_aoi.npy, data/2026_RGBNIR_aoi.npy,
        data/worldcover2021_aoi_remapped.npy
Writes: figures/worldcover_validation.png

1. ESA WorldCover 2021 -- the source of the training labels (model-derived, with its
   own errors). Grid-resized to the model's array, not CRS-reprojected: approximate.
   Reports overall pixel agreement and per-class IoU.
2. NDWI > 0 as a reference water mask. Shadows (incl. cloud shadow) can also have
   NDWI > 0. Reports precision, recall and IoU for the Water/wetland class.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


class CFG:
    DATA_DIR = Path("data")
    FIG_DIR = Path("figures")

    DATE1_LABEL, DATE2_LABEL = "2021", "2026"
    DATE1_AOI_PATH = DATA_DIR / "2021_RGBNIR_aoi.npy"
    DATE2_AOI_PATH = DATA_DIR / "2026_RGBNIR_aoi.npy"
    DATE1_LC_PATH = DATA_DIR / "2021_landcover.npy"
    DATE2_LC_PATH = DATA_DIR / "2026_landcover.npy"
    WORLDCOVER_PATH = DATA_DIR / "worldcover2021_aoi_remapped.npy"
    FIG_PATH = FIG_DIR / "worldcover_validation.png"

    NUM_CLASSES = 7
    PIXELS_PER_KM2 = 10_000
    CLASS_NAMES = ["Ignore", "Tree cover", "Low vegetation", "Cropland",
                   "Built-up", "Bare/sparse", "Water/wetland"]
    CLASS_COLORS = ["#FFFFFF", "#1A6B1A", "#8FCB6E", "#E8C24A",
                    "#C0392B", "#B08D57", "#3B82C4"]
    WATER_ID = 6


def worldcover_check(model_lc, wc_lc):
    valid = (model_lc != 0) & (wc_lc != 0)
    agreement = (model_lc == wc_lc)[valid].sum() / valid.sum() * 100
    print(f"WorldCover overall pixel agreement ({CFG.DATE1_LABEL}): {agreement:.1f}%")

    pred, ref = model_lc[valid], wc_lc[valid]
    print("Per-class IoU vs WorldCover (%):")
    ious = {}
    for cid in range(1, CFG.NUM_CLASSES):
        tp = ((pred == cid) & (ref == cid)).sum()
        union = ((pred == cid) | (ref == cid)).sum()
        ious[CFG.CLASS_NAMES[cid]] = tp / union * 100 if union > 0 else float("nan")
        print(f"  {CFG.CLASS_NAMES[cid]:16s} {ious[CFG.CLASS_NAMES[cid]]:5.1f}")
    return agreement, valid


def save_worldcover_figure(model_lc, wc_lc, valid, agreement):
    CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(CFG.CLASS_COLORS)
    norm = BoundaryNorm(list(range(len(CFG.CLASS_COLORS) + 1)), cmap.N)
    handles = [Patch(facecolor=CFG.CLASS_COLORS[i], label=CFG.CLASS_NAMES[i]) for i in range(1, 7)]

    agree_map = np.zeros(model_lc.shape, dtype=np.uint8)          # 0 = not compared
    agree_map[valid & (model_lc == wc_lc)] = 1                    # 1 = agree
    agree_map[valid & (model_lc != wc_lc)] = 2                    # 2 = disagree

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    axes[0].imshow(model_lc, cmap=cmap, norm=norm)
    axes[0].set_title(f"Model ({CFG.DATE1_LABEL})")
    axes[1].imshow(wc_lc, cmap=cmap, norm=norm)
    axes[1].set_title("ESA WorldCover 2021 (remapped)")
    axes[2].imshow(agree_map, cmap=ListedColormap(["#FFFFFF", "#6BBF59", "#D64541"]), vmin=0, vmax=2)
    axes[2].set_title(f"Agreement: {agreement:.1f}% of pixels (green = agree, red = disagree)")
    for ax in axes:
        ax.axis("off")
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.savefig(CFG.FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", CFG.FIG_PATH)


def compute_ndwi(rgbnir_uint8):
    img = rgbnir_uint8.astype(np.float32)
    green, nir = img[..., 1], img[..., 3]
    return (green - nir) / (green + nir + 1e-6)


def ndwi_check(label, rgbnir, lc):
    ndwi_water = compute_ndwi(rgbnir) > 0.0
    model_water = lc == CFG.WATER_ID
    tp = (model_water & ndwi_water).sum()
    precision = tp / model_water.sum() if model_water.sum() > 0 else float("nan")
    recall = tp / ndwi_water.sum() if ndwi_water.sum() > 0 else float("nan")
    union = (model_water | ndwi_water).sum()
    iou = tp / union if union > 0 else float("nan")
    print(f"{label}: model water {model_water.sum() / CFG.PIXELS_PER_KM2:.2f} km^2, "
          f"NDWI water {ndwi_water.sum() / CFG.PIXELS_PER_KM2:.2f} km^2 | "
          f"precision {precision:.2f}, recall {recall:.2f}, IoU {iou:.2f}")


def main():
    lc1 = np.load(CFG.DATE1_LC_PATH)
    lc2 = np.load(CFG.DATE2_LC_PATH)

    # ---- 1. WorldCover
    if CFG.WORLDCOVER_PATH.exists():
        wc_lc = np.load(CFG.WORLDCOVER_PATH)
        if wc_lc.shape != lc1.shape:
            raise SystemExit(f"Shape mismatch: WorldCover {wc_lc.shape} vs model map {lc1.shape}. "
                             "Resize/reproject the reference to the model grid first.")
        agreement, valid = worldcover_check(lc1, wc_lc)
        save_worldcover_figure(lc1, wc_lc, valid, agreement)
    else:
        print(f"WorldCover reference not found ({CFG.WORLDCOVER_PATH}) -- skipping this check.")

    # ---- 2. NDWI
    print()
    for label, img_path, lc in [(CFG.DATE1_LABEL, CFG.DATE1_AOI_PATH, lc1),
                                (CFG.DATE2_LABEL, CFG.DATE2_AOI_PATH, lc2)]:
        ndwi_check(label, np.load(img_path), lc)


if __name__ == "__main__":
    main()
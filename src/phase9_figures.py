"""Phase 9 -- all report figures (fig1 to fig8, 10 PNG files).

Run from the repo root:  python phase9_figures.py
Reads : data/2021_RGBNIR_aoi.npy, data/2026_RGBNIR_aoi.npy, data/*_landcover.npy,
        data/nonbuiltup_to_builtup_mask.npy, data/hotspot_grid.npy
Writes: figures/fig1_study_area.png, fig2_segmentation.png,
        fig3_builtup_2021.png, fig3_builtup_2026.png,
        fig4_vegetation_2021.png, fig4_vegetation_2026.png,
        fig5_comparison.png, fig6_change_detection.png,
        fig7_expansion_map.png, fig8_hotspots.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle


class CFG:
    DATA_DIR = Path("data")
    FIG_DIR = Path("figures")

    DATE1_LABEL, DATE2_LABEL = "2021", "2026"
    DATE1_AOI_PATH = DATA_DIR / "2021_RGBNIR_aoi.npy"
    DATE2_AOI_PATH = DATA_DIR / "2026_RGBNIR_aoi.npy"
    DATE1_LC_PATH = DATA_DIR / "2021_landcover.npy"
    DATE2_LC_PATH = DATA_DIR / "2026_landcover.npy"
    EXPANSION_MASK_PATH = DATA_DIR / "nonbuiltup_to_builtup_mask.npy"
    HOTSPOT_GRID_PATH = DATA_DIR / "hotspot_grid.npy"

    PIXELS_PER_KM2 = 10_000
    CELL_SIZE_PX = 50
    IGNORE_ID = 0
    GREEN_IDS = (1, 2)
    BUILTUP_ID = 4

    AOI_NAME = "Khirdalan - Masazir - Binagadi, Azerbaijan"
    CLASS_NAMES = ["Ignore", "Tree cover", "Low vegetation", "Cropland",
                   "Built-up", "Bare/sparse", "Water/wetland"]
    CLASS_COLORS = ["#FFFFFF", "#1A6B1A", "#8FCB6E", "#E8C24A",
                    "#C0392B", "#B08D57", "#3B82C4"]


def landcover_cmap_norm():
    cmap = ListedColormap(CFG.CLASS_COLORS)
    norm = BoundaryNorm(list(range(len(CFG.CLASS_COLORS) + 1)), cmap.N)
    return cmap, norm


def save_fig(fig, name):
    path = CFG.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def highlight_fig(landcover, class_ids, color, name, year):
    """One class group highlighted on a light-grey background (figures 3 and 4)."""
    mask = np.isin(landcover, class_ids)
    km2 = mask.sum() / CFG.PIXELS_PER_KM2
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.zeros(landcover.shape), cmap=ListedColormap(["#E6E6E6"]), vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]), vmin=0, vmax=1)
    ax.set_title(f"{name} -- {year} ({mask.mean() * 100:.1f}% of area, {km2:.1f} km^2)")
    ax.axis("off")
    return fig


def change_detection_fig(lc1, lc2):
    """Per-pixel change categories between the two dates (figure 6)."""
    valid = (lc1 != CFG.IGNORE_ID) & (lc2 != CFG.IGNORE_ID)
    changed = (lc1 != lc2) & valid
    veg_to_built = np.isin(lc1, CFG.GREEN_IDS) & (lc2 == CFG.BUILTUP_ID)
    crop_to_built = (lc1 == 3) & (lc2 == CFG.BUILTUP_ID)
    bare_to_built = (lc1 == 5) & (lc2 == CFG.BUILTUP_ID)
    other_change = changed & ~veg_to_built & ~crop_to_built & ~bare_to_built

    change_map = np.zeros(lc1.shape, dtype=np.uint8)
    change_map[other_change] = 4
    change_map[bare_to_built] = 3
    change_map[crop_to_built] = 2
    change_map[veg_to_built] = 1

    colors = ["#D9D9D9", "#F39C12", "#C0392B", "#8E5A2B", "#95A5A6"]
    names = ["Unchanged", "Vegetation -> Built-up", "Cropland -> Built-up",
             "Bare -> Built-up", "Other change (unverified; may include noise)"]
    areas = [(valid & ~changed).sum(), veg_to_built.sum(), crop_to_built.sum(),
             bare_to_built.sum(), other_change.sum()]

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)
    ax.imshow(np.ma.masked_where(~valid, change_map), cmap=ListedColormap(colors), norm=norm)
    ax.set_title(f"Change Detection: {CFG.DATE1_LABEL} -> {CFG.DATE2_LABEL}")
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=c, label=f"{n} ({a / CFG.PIXELS_PER_KM2:.2f} km^2)")
                       for c, n, a in zip(colors, names, areas)],
              loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    return fig


def main():
    CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)

    date1_crop = np.load(CFG.DATE1_AOI_PATH)
    date2_crop = np.load(CFG.DATE2_AOI_PATH)
    date1_landcover = np.load(CFG.DATE1_LC_PATH)
    date2_landcover = np.load(CFG.DATE2_LC_PATH)
    expansion_mask = np.load(CFG.EXPANSION_MASK_PATH)
    hotspot_grid = np.load(CFG.HOTSPOT_GRID_PATH)

    cmap, norm = landcover_cmap_norm()
    legend_handles = [Patch(facecolor=CFG.CLASS_COLORS[i], label=CFG.CLASS_NAMES[i]) for i in range(1, 7)]

    # Figure 1 -- study area
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(date1_crop[..., :3])
    ax.set_title(f"Study Area -- {CFG.AOI_NAME}")
    save_fig(fig, "fig1_study_area.png")

    # Figure 2 -- segmentation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(date2_landcover, cmap=cmap, norm=norm)
    ax.set_title(f"Land-Cover Segmentation -- {CFG.DATE2_LABEL}")
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    save_fig(fig, "fig2_segmentation.png")

    # Figure 3 -- built-up, both dates
    for label, lc in [(CFG.DATE1_LABEL, date1_landcover), (CFG.DATE2_LABEL, date2_landcover)]:
        save_fig(highlight_fig(lc, [CFG.BUILTUP_ID], "#C0392B", "Built-up", label),
                 f"fig3_builtup_{label}.png")

    # Figure 4 -- vegetation (tree cover + low vegetation), both dates
    for label, lc in [(CFG.DATE1_LABEL, date1_landcover), (CFG.DATE2_LABEL, date2_landcover)]:
        save_fig(highlight_fig(lc, CFG.GREEN_IDS, "#2E8B2E", "Vegetation (tree cover + low vegetation)", label),
                 f"fig4_vegetation_{label}.png")

    # Figure 5 -- side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(date1_landcover, cmap=cmap, norm=norm)
    axes[0].set_title(CFG.DATE1_LABEL)
    axes[1].imshow(date2_landcover, cmap=cmap, norm=norm)
    axes[1].set_title(CFG.DATE2_LABEL)
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle("Land-Cover Comparison", fontsize=14)
    save_fig(fig, "fig5_comparison.png")

    # Figure 6 -- change detection
    save_fig(change_detection_fig(date1_landcover, date2_landcover), "fig6_change_detection.png")

    # Figure 7 -- urban expansion
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(date2_crop[..., :3])
    ax.imshow(np.ma.masked_where(~expansion_mask, expansion_mask), cmap=ListedColormap(["#FF3B30"]), alpha=0.75)
    ax.set_title(f"Urban Expansion {CFG.DATE1_LABEL}-{CFG.DATE2_LABEL} "
                 f"({expansion_mask.sum() / CFG.PIXELS_PER_KM2:.2f} km^2 gross)")
    save_fig(fig, "fig7_expansion_map.png")

    # Figure 8 -- hotspots
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(date2_crop[..., :3])
    max_count = hotspot_grid.max() if hotspot_grid.max() > 0 else 1
    for r in range(hotspot_grid.shape[0]):
        for c in range(hotspot_grid.shape[1]):
            val = hotspot_grid[r, c]
            if val == 0:
                continue
            intensity = float(np.clip(val / max_count, 0.1, 0.9))
            y0, x0 = r * CFG.CELL_SIZE_PX, c * CFG.CELL_SIZE_PX
            ax.add_patch(Rectangle((x0, y0), CFG.CELL_SIZE_PX, CFG.CELL_SIZE_PX,
                                   facecolor="#FF3B30", alpha=intensity, edgecolor="none"))
    ax.set_title("Urban Development Hotspots")
    save_fig(fig, "fig8_hotspots.png")

    print("\nAll figures saved to", CFG.FIG_DIR.resolve())


if __name__ == "__main__":
    main()
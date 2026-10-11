"""Phase 7 -- development hotspots on a 500 m grid.

Run from the repo root:  python phase7_hotspot.py
Reads : data/nonbuiltup_to_builtup_mask.npy
Writes: data/hotspot_grid.npy
"""
from pathlib import Path

import numpy as np


class CFG:
    DATA_DIR = Path("data")
    EXPANSION_MASK_PATH = DATA_DIR / "nonbuiltup_to_builtup_mask.npy"
    HOTSPOT_GRID_PATH = DATA_DIR / "hotspot_grid.npy"

    PIXELS_PER_KM2 = 10_000
    CELL_SIZE_PX = 50   # hotspot grid cell size (500 m at 10 m/px)


def build_hotspot_grid(expansion_mask, cell_size=CFG.CELL_SIZE_PX):
    h, w = expansion_mask.shape
    pad_h = (cell_size - h % cell_size) % cell_size
    pad_w = (cell_size - w % cell_size) % cell_size
    padded = np.pad(expansion_mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    grid_h, grid_w = padded.shape[0] // cell_size, padded.shape[1] // cell_size
    return padded.reshape(grid_h, cell_size, grid_w, cell_size).sum(axis=(1, 3)).astype(np.int64)


def main():
    expansion_mask = np.load(CFG.EXPANSION_MASK_PATH)
    hotspot_grid = build_hotspot_grid(expansion_mask)
    np.save(CFG.HOTSPOT_GRID_PATH, hotspot_grid)

    flat_idx = np.argsort(hotspot_grid.flatten())[::-1][:10]
    rows, cols = np.unravel_index(flat_idx, hotspot_grid.shape)
    cell_px = CFG.CELL_SIZE_PX ** 2

    print("Top 10 hotspot cells (500m x 500m):")
    for r, c in zip(rows, cols):
        px = hotspot_grid[r, c]
        if px == 0:
            continue
        print(f"  rows {r * CFG.CELL_SIZE_PX}-{(r + 1) * CFG.CELL_SIZE_PX}, "
              f"cols {c * CFG.CELL_SIZE_PX}-{(c + 1) * CFG.CELL_SIZE_PX}: "
              f"{px / CFG.PIXELS_PER_KM2:.4f} km^2 ({px / cell_px * 100:.0f}% of the cell converted)")

    total = hotspot_grid.sum()
    top10 = np.sort(hotspot_grid.flatten())[::-1][:10].sum()
    print(f"\nTop-10 cells hold {top10 / total * 100:.1f}% of the gross expansion "
          f"({top10 / CFG.PIXELS_PER_KM2:.2f} of {total / CFG.PIXELS_PER_KM2:.2f} km^2)")
    print("Saved:", CFG.HOTSPOT_GRID_PATH)


if __name__ == "__main__":
    main()
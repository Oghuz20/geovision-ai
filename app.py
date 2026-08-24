"""
GeoVision AI -- Interactive Decision-Support Dashboard

Run with:
    streamlit run app.py

Reads all previously computed outputs (no model inference happens here --
this is a viewer over what Phases 4-8 already produced):
    data/2021_RGBNIR_aoi.npy, data/2026_RGBNIR_aoi.npy
    data/2021_landcover.npy, data/2026_landcover.npy
    data/nonbuiltup_to_builtup_mask.npy
    data/hotspot_grid.npy

If any file is missing, that section of the dashboard shows a clear
message instead of crashing -- lets you demo whatever's ready even if
the full pipeline hasn't been rerun since the last model swap.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle


# =================================================================
# Config
# =================================================================
class CFG:
    DATA_DIR = Path("data")

    DATE1_RGBNIR_PATH = DATA_DIR / "2021_RGBNIR_aoi.npy"
    DATE2_RGBNIR_PATH = DATA_DIR / "2026_RGBNIR_aoi.npy"
    DATE1_LC_PATH = DATA_DIR / "2021_landcover.npy"
    DATE2_LC_PATH = DATA_DIR / "2026_landcover.npy"
    EXPANSION_MASK_PATH = DATA_DIR / "nonbuiltup_to_builtup_mask.npy"
    HOTSPOT_GRID_PATH = DATA_DIR / "hotspot_grid.npy"

    PIXELS_PER_KM2 = 10_000
    CELL_SIZE_PX = 50

    CLASS_NAMES = ["Ignore", "Tree cover", "Low vegetation", "Cropland",
                   "Built-up", "Bare/sparse", "Water/wetland"]
    CLASS_COLORS = ["#FFFFFF", "#1A6B1A", "#8FCB6E", "#E8C24A",
                     "#C0392B", "#B08D57", "#3B82C4"]

    GREEN_IDS = (1, 2)
    BUILTUP_ID = 4
    IGNORE_ID = 0

    AOI_BOUNDS = (49.73, 40.41, 49.87, 40.53)   # min_lon, min_lat, max_lon, max_lat
    AOI_NAME = "Khirdalan - Masazir - Binagadi, Azerbaijan"

    # --- Validation results (from the held-out test set + real-AOI cross-checks) ---
    # Copied from the actual evaluation runs -- update if a newer model is adopted.
    TEST_MEAN_IOU = 0.6111
    TEST_FWIOU = 0.6838
    TEST_PRECISION = 0.7584
    TEST_RECALL = 0.7706
    TEST_F1 = 0.7385

    WORLDCOVER_OVERALL_AGREEMENT = 71.0
    WORLDCOVER_PER_CLASS = {
        "Tree cover": 5.8, "Low vegetation": 37.3, "Cropland": 12.1,
        "Built-up": 77.9, "Bare/sparse": 19.4, "Water/wetland": 65.8,
    }

    WORLDCOVER_FIG_PATH = Path("figures") / "worldcover_validation.png"
    NDWI_FIG_PATHS = {
        "2021": Path("figures") / "water_sanity_check_2021.png",
        "2026": Path("figures") / "water_sanity_check_2026.png",
    }


# =================================================================
# Data loading (cached so switching tabs doesn't reload from disk)
# =================================================================
@st.cache_data
def load_array(path: Path):
    if not path.exists():
        return None
    return np.load(path)


def landcover_cmap_norm():
    cmap = ListedColormap(CFG.CLASS_COLORS)
    norm = BoundaryNorm(list(range(len(CFG.CLASS_COLORS) + 1)), cmap.N)
    return cmap, norm


def landcover_legend_handles():
    return [Patch(facecolor=CFG.CLASS_COLORS[i], label=CFG.CLASS_NAMES[i])
            for i in range(1, len(CFG.CLASS_NAMES))]


# =================================================================
# Stats (recomputed live from the loaded arrays -- always in sync,
# never hardcoded, so the dashboard reflects whichever checkpoint's
# output currently sits in data/)
# =================================================================
def compute_area_stats(landcover: np.ndarray) -> dict:
    counts = np.bincount(landcover.flatten(), minlength=7)
    valid = counts[1:].sum()

    stats = {}
    for i in range(1, 7):
        km2 = counts[i] / CFG.PIXELS_PER_KM2
        pct = (counts[i] / valid * 100) if valid > 0 else 0.0
        stats[CFG.CLASS_NAMES[i]] = {"km2": km2, "pct": pct}

    green_px = sum(counts[c] for c in CFG.GREEN_IDS)
    stats["Green space"] = {
        "km2": green_px / CFG.PIXELS_PER_KM2,
        "pct": (green_px / valid * 100) if valid > 0 else 0.0,
    }
    return stats


def compute_planning_indicators(stats1: dict, stats2: dict, expansion_mask, hotspot_grid) -> list:
    indicators = []

    green_pct = stats2["Green space"]["pct"]
    status = "OK" if green_pct >= 15.0 else "FLAG"
    indicators.append(("Green space ratio", status, f"{green_pct:.1f}%",
                        "Above the 15% reference level." if status == "OK"
                        else "Below the 15% reference level -- potential loss of green areas."))

    crop1, crop2 = stats1["Cropland"]["km2"], stats2["Cropland"]["km2"]
    loss_pct = (crop1 - crop2) / crop1 * 100 if crop1 > 0 else 0.0
    status = "FLAG" if loss_pct >= 25.0 else "WATCH"
    indicators.append(("Agricultural land loss", status, f"{loss_pct:.0f}%",
                        "Notable conversion of agricultural land." if status == "FLAG"
                        else "Below the 25% flag threshold."))

    builtup1, builtup2 = stats1["Built-up"]["pct"], stats2["Built-up"]["pct"]
    status = "FLAG" if builtup2 >= 60 else ("WATCH" if builtup2 >= 55 else "OK")
    indicators.append(("Built-up saturation", status, f"{builtup2:.1f}%",
                        f"Growth of {builtup2 - builtup1:+.1f} points since the earlier date."))

    if expansion_mask is not None and hotspot_grid is not None:
        total_expansion_km2 = expansion_mask.sum() / CFG.PIXELS_PER_KM2
        top10_km2 = np.sort(hotspot_grid.flatten())[::-1][:10].sum() / CFG.PIXELS_PER_KM2
        diffuse_pct = (total_expansion_km2 - top10_km2) / total_expansion_km2 * 100 if total_expansion_km2 > 0 else 0
        status = "FLAG" if diffuse_pct >= 70 else "OK"
        indicators.append(("Development pattern", status, f"{diffuse_pct:.0f}% diffuse",
                            "Growth is spread broadly with a few concentrated hotspots -- "
                            "potential sprawl/fragmentation." if status == "FLAG"
                            else "Development is relatively concentrated."))

    return indicators


# =================================================================
# Plot builders (return a Figure -- no file I/O, unlike phase9_figures.py)
# =================================================================
def plot_rgb(rgbnir, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgbnir[..., :3])
    ax.set_title(title)
    ax.axis("off")
    return fig


def plot_landcover(landcover, title):
    cmap, norm = landcover_cmap_norm()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(landcover, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=landcover_legend_handles(), loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=8)
    return fig


def plot_class_highlight(landcover, class_ids, color, title):
    mask = np.isin(landcover, class_ids)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.ones_like(landcover), cmap="Greys", vmin=0, vmax=2)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]))
    ax.set_title(title)
    ax.axis("off")
    return fig


def plot_change_detection(lc1, lc2):
    valid = (lc1 != CFG.IGNORE_ID) & (lc2 != CFG.IGNORE_ID)
    changed = (lc1 != lc2) & valid
    veg_to_built = np.isin(lc1, CFG.GREEN_IDS) & (lc2 == CFG.BUILTUP_ID)
    cropland_to_built = (lc1 == 3) & (lc2 == CFG.BUILTUP_ID)
    bare_to_built = (lc1 == 5) & (lc2 == CFG.BUILTUP_ID)
    other_change = changed & ~veg_to_built & ~cropland_to_built & ~bare_to_built

    change_map = np.zeros(lc1.shape, dtype=np.uint8)
    change_map[other_change] = 4
    change_map[bare_to_built] = 3
    change_map[cropland_to_built] = 2
    change_map[veg_to_built] = 1

    colors = ["#D9D9D9", "#F39C12", "#C0392B", "#8E5A2B", "#95A5A6"]
    labels = ["Unchanged", "Vegetation -> Built-up", "Cropland -> Built-up",
              "Bare -> Built-up", "Other change (noise/boundary)"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.ma.masked_where(~valid, change_map), cmap=ListedColormap(colors), vmin=0, vmax=4)
    ax.set_title("Change Detection: 2021 -> 2026")
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)],
              loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    return fig


def plot_expansion(expansion_mask, rgbnir_base):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgbnir_base[..., :3])
    overlay = np.ma.masked_where(~expansion_mask, expansion_mask)
    ax.imshow(overlay, cmap=ListedColormap(["#FF3B30"]), alpha=0.75)
    km2 = expansion_mask.sum() / CFG.PIXELS_PER_KM2
    ax.set_title(f"Urban Expansion (Non-Built-up -> Built-up)\nTotal: {km2:.2f} km^2")
    ax.axis("off")
    return fig


def plot_hotspots(hotspot_grid, rgbnir_base, cell_size=CFG.CELL_SIZE_PX):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgbnir_base[..., :3])
    max_count = hotspot_grid.max() if hotspot_grid.max() > 0 else 1
    for r in range(hotspot_grid.shape[0]):
        for c in range(hotspot_grid.shape[1]):
            val = hotspot_grid[r, c]
            if val == 0:
                continue
            intensity = float(np.clip(val / max_count, 0.1, 0.9))
            y0, x0 = r * cell_size, c * cell_size
            ax.add_patch(Rectangle((x0, y0), cell_size, cell_size,
                                    facecolor="#FF3B30", alpha=intensity, edgecolor="none"))
    ax.set_title("Urban Development Hotspots")
    ax.axis("off")
    return fig


def fig_download_button(fig, filename: str, label: str = "Download this figure (PNG)"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/png")


# =================================================================
# App
# =================================================================
def main():
    st.set_page_config(page_title="GeoVision AI -- Urban Planning Support", layout="wide")
    st.title("GeoVision AI")
    st.caption(f"AI-based Urban Area Analysis & Planning Decision Support -- {CFG.AOI_NAME}")

    rgbnir1 = load_array(CFG.DATE1_RGBNIR_PATH)
    rgbnir2 = load_array(CFG.DATE2_RGBNIR_PATH)
    lc1 = load_array(CFG.DATE1_LC_PATH)
    lc2 = load_array(CFG.DATE2_LC_PATH)
    expansion_mask = load_array(CFG.EXPANSION_MASK_PATH)
    hotspot_grid = load_array(CFG.HOTSPOT_GRID_PATH)

    if lc1 is None or lc2 is None:
        st.error("Land-cover outputs not found in data/. Run Phase 4 (segmentation) first.")
        return

    tab_overview, tab_explorer, tab_change, tab_hotspots, tab_indicators, tab_validation = st.tabs(
        ["Overview", "Land Cover Explorer", "Change Detection", "Development Hotspots",
         "Planning Indicators", "Model Validation"]
    )

    stats1 = compute_area_stats(lc1)
    stats2 = compute_area_stats(lc2)

    # --- Overview ---
    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            if rgbnir1 is not None:
                st.pyplot(plot_rgb(rgbnir1, "Study Area -- 2021"))
        with col2:
            st.subheader("Key Metrics (2021 -> 2026)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Built-up", f"{stats2['Built-up']['pct']:.1f}%",
                      f"{stats2['Built-up']['pct'] - stats1['Built-up']['pct']:+.1f} pts")
            m2.metric("Green space", f"{stats2['Green space']['pct']:.1f}%",
                      f"{stats2['Green space']['pct'] - stats1['Green space']['pct']:+.1f} pts")
            m3.metric("Cropland", f"{stats2['Cropland']['pct']:.1f}%",
                      f"{stats2['Cropland']['pct'] - stats1['Cropland']['pct']:+.1f} pts")

            if expansion_mask is not None:
                st.metric("Total urban expansion", f"{expansion_mask.sum() / CFG.PIXELS_PER_KM2:.2f} km^2")

            st.subheader("Land-use balance, 2026")
            df = pd.DataFrame({
                "Class": list(stats2.keys()),
                "Area (km^2)": [round(v["km2"], 2) for v in stats2.values()],
                "Share (%)": [round(v["pct"], 1) for v in stats2.values()],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download this table (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="landuse_balance_2026.csv",
                mime="text/csv",
            )

    # --- Land Cover Explorer ---
    with tab_explorer:
        date_choice = st.radio("Date", ["2021", "2026"], horizontal=True)
        lc = lc1 if date_choice == "2021" else lc2
        rgbnir = rgbnir1 if date_choice == "2021" else rgbnir2

        layer = st.selectbox("Layer", ["Land Cover Map", "Built-up Highlight", "Vegetation Highlight", "True Color"])

        if layer == "Land Cover Map":
            fig = plot_landcover(lc, f"Land Cover -- {date_choice}")
            st.pyplot(fig)
            fig_download_button(fig, f"landcover_{date_choice}.png")
        elif layer == "Built-up Highlight":
            fig = plot_class_highlight(lc, [CFG.BUILTUP_ID], "#C0392B", f"Built-up -- {date_choice}")
            st.pyplot(fig)
            fig_download_button(fig, f"builtup_{date_choice}.png")
        elif layer == "Vegetation Highlight":
            fig = plot_class_highlight(lc, CFG.GREEN_IDS, "#2E8B2E", f"Vegetation -- {date_choice}")
            st.pyplot(fig)
            fig_download_button(fig, f"vegetation_{date_choice}.png")
        elif layer == "True Color" and rgbnir is not None:
            fig = plot_rgb(rgbnir, f"True Color -- {date_choice}")
            st.pyplot(fig)
            fig_download_button(fig, f"truecolor_{date_choice}.png")

    # --- Change Detection ---
    with tab_change:
        fig_change = plot_change_detection(lc1, lc2)
        st.pyplot(fig_change)
        fig_download_button(fig_change, "change_detection.png")

        if expansion_mask is not None and rgbnir2 is not None:
            fig_exp = plot_expansion(expansion_mask, rgbnir2)
            st.pyplot(fig_exp)
            fig_download_button(fig_exp, "urban_expansion.png")

    # --- Development Hotspots ---
    with tab_hotspots:
        if hotspot_grid is not None and rgbnir2 is not None:
            fig_hot = plot_hotspots(hotspot_grid, rgbnir2)
            st.pyplot(fig_hot)
            fig_download_button(fig_hot, "development_hotspots.png")
        else:
            st.info("Hotspot data not found -- run Phase 7 first.")

    # --- Model Validation ---
    with tab_validation:
        st.subheader("Held-out test set performance")
        st.caption("Measured on labeled Sentinel-2 tiles never seen during training -- "
                   "not this study area specifically (see below for that check).")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean IoU", f"{CFG.TEST_MEAN_IOU:.3f}")
        c2.metric("FWIoU", f"{CFG.TEST_FWIOU:.3f}")
        c3.metric("Precision", f"{CFG.TEST_PRECISION:.3f}")
        c4.metric("Recall", f"{CFG.TEST_RECALL:.3f}")
        c5.metric("F1", f"{CFG.TEST_F1:.3f}")

        st.divider()

        st.subheader("Real-AOI cross-validation vs. ESA WorldCover 2021")
        st.caption("WorldCover 2021 is the same source dataset the training labels were built "
                   "from -- this checks generalization to this specific, previously-unseen area. "
                   "Note: grid-resized for comparison rather than fully CRS-reprojected, so this "
                   "is an approximate agreement check, not a survey-grade confusion matrix.")
        st.metric("Overall agreement", f"{CFG.WORLDCOVER_OVERALL_AGREEMENT:.1f}%")

        wc_df = pd.DataFrame({
            "Class": list(CFG.WORLDCOVER_PER_CLASS.keys()),
            "Agreement (%)": list(CFG.WORLDCOVER_PER_CLASS.values()),
        })
        st.dataframe(wc_df, use_container_width=True, hide_index=True)

        if CFG.WORLDCOVER_FIG_PATH.exists():
            st.image(str(CFG.WORLDCOVER_FIG_PATH), use_container_width=True)
        else:
            st.info("WorldCover comparison figure not found -- run worldcover_validation.py first.")

        st.divider()

        st.subheader("Water-class sanity check (NDWI)")
        st.caption("Independent, physics-based cross-check for the Water/wetland class -- "
                   "no training data involved.")
        ndwi_cols = st.columns(2)
        for col, (date_label, path) in zip(ndwi_cols, CFG.NDWI_FIG_PATHS.items()):
            with col:
                if path.exists():
                    st.image(str(path), caption=date_label, use_container_width=True)
                else:
                    st.info(f"NDWI figure for {date_label} not found -- run water_sanity_check.py first.")

        st.divider()

        st.subheader("Known limitations")
        st.markdown(
            "- **Bare/sparse vs. Low vegetation boundary confusion**: the WorldCover comparison "
            "above confirms the model systematically over-predicts Bare/sparse where the reference "
            "data shows Low vegetation.\n"
            "- **Water-extent changes near the northern lake** may be partly attributable to cloud "
            "shadow in the 2026 source imagery rather than genuine water-level change -- see the "
            "NDWI check above.\n"
            "- **The held-out test set's geographic tile composition** underrepresents Built-up "
            "and Cropland relative to train/validation, so the headline test metrics likely "
            "understate true performance on those two classes somewhat.\n"
            "- **Class-transition noise** (e.g. isolated Built-up <-> vegetation flips between "
            "dates) reflects pixel-boundary classification noise, not real land-use change -- "
            "handled explicitly as a separate category in the Change Detection tab."
        )

    # --- Planning Indicators ---
    with tab_indicators:
        indicators = compute_planning_indicators(stats1, stats2, expansion_mask, hotspot_grid)
        for name, status, value, message in indicators:
            if status == "FLAG":
                st.error(f"**{name}**: {value} -- {message}")
            elif status == "WATCH":
                st.warning(f"**{name}**: {value} -- {message}")
            else:
                st.success(f"**{name}**: {value} -- {message}")


if __name__ == "__main__":
    main()

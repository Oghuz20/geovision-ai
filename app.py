"""
GeoVision AI -- Interactive Decision-Support Dashboard

Run with:
    streamlit run app.py

Reads all previously computed outputs (no model inference happens here --
this is a viewer over what notebooks/02_deployment_and_analysis.ipynb already produced):
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
    WATER_ID = 6
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
    CROPLAND_TEST_IOU = 0.34
    TEST_BUILTUP_DOMINANT_PATCHES = 44      # of 13,407 test patches

    WORLDCOVER_OVERALL_AGREEMENT = 71.0     # pixel agreement, %
    WORLDCOVER_PER_CLASS = {                # per-class IoU vs WorldCover, %
        "Tree cover": 5.8, "Low vegetation": 37.3, "Cropland": 12.1,
        "Built-up": 77.9, "Bare/sparse": 19.4, "Water/wetland": 65.8,
    }

    WORLDCOVER_FIG_PATH = Path("figures") / "worldcover_validation.png"


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


def built_up_flows(lc1: np.ndarray, lc2: np.ndarray):
    """Gross gain, reverse flow and net change of built-up area, in km^2."""
    valid = (lc1 != CFG.IGNORE_ID) & (lc2 != CFG.IGNORE_ID)
    b1, b2 = lc1 == CFG.BUILTUP_ID, lc2 == CFG.BUILTUP_ID
    gain = (b2 & ~b1 & valid).sum() / CFG.PIXELS_PER_KM2
    loss = (b1 & ~b2 & valid).sum() / CFG.PIXELS_PER_KM2
    return gain, loss, gain - loss


def ndwi_water_check(rgbnir: np.ndarray, lc: np.ndarray) -> dict:
    """Consistency of the model's Water/wetland class with an NDWI > 0 water mask.

    NDWI = (Green - NIR) / (Green + NIR) on the uint8-scaled bands. NDWI is a
    reference index, not ground truth: shadows (including cloud shadow) can also
    have NDWI > 0. Returns precision, recall and IoU of model-water vs NDWI-water.
    """
    img = rgbnir.astype(np.float32)
    green, nir = img[..., 1], img[..., 3]
    ref = ((green - nir) / (green + nir + 1e-6)) > 0.0
    model = lc == CFG.WATER_ID
    tp = float((model & ref).sum())
    n_model, n_ref, union = float(model.sum()), float(ref.sum()), float((model | ref).sum())
    return {
        "precision": tp / n_model if n_model else float("nan"),
        "recall": tp / n_ref if n_ref else float("nan"),
        "iou": tp / union if union else float("nan"),
        "model_km2": n_model / CFG.PIXELS_PER_KM2,
        "ndwi_km2": n_ref / CFG.PIXELS_PER_KM2,
    }


def compute_planning_indicators(stats1: dict, stats2: dict, expansion_mask, hotspot_grid,
                                lc1=None, lc2=None) -> list:
    indicators = []

    green_pct = stats2["Green space"]["pct"]
    status = "OK" if green_pct >= 15.0 else "FLAG"
    indicators.append(("Vegetated cover (tree + low vegetation)", status, f"{green_pct:.1f}%",
                        "Above the 15% illustrative reference level. This includes semi-desert "
                        "scrub, not only urban green space." if status == "OK"
                        else "Below the 15% illustrative reference level -- potential loss of "
                             "vegetated cover."))

    crop1, crop2 = stats1["Cropland"]["km2"], stats2["Cropland"]["km2"]
    loss_pct = (crop1 - crop2) / crop1 * 100 if crop1 > 0 else 0.0
    status = "FLAG" if loss_pct >= 25.0 else "WATCH"
    if status == "FLAG":
        message = f"Mapped cropland fell by {crop1 - crop2:.2f} km^2."
        if lc1 is not None and lc2 is not None:
            to_built = ((lc1 == 3) & (lc2 == CFG.BUILTUP_ID)).sum() / CFG.PIXELS_PER_KM2
            to_lowveg = ((lc1 == 3) & (lc2 == 2)).sum() / CFG.PIXELS_PER_KM2
            message += (f" Of this, {to_built:.2f} km^2 became built-up and "
                        f"{to_lowveg:.2f} km^2 became low vegetation (a known confusion pair).")
        message += (f" Cropland test IoU is only {CFG.CROPLAND_TEST_IOU:.2f}, so treat this as a "
                    "monitoring signal, not a confirmed loss.")
    else:
        message = "Below the 25% flag threshold."
    indicators.append(("Agricultural land loss", status, f"{loss_pct:.0f}%", message))

    builtup1, builtup2 = stats1["Built-up"]["pct"], stats2["Built-up"]["pct"]
    status = "FLAG" if builtup2 >= 60 else ("WATCH" if builtup2 >= 55 else "OK")
    indicators.append(("Built-up saturation", status, f"{builtup2:.1f}%",
                        f"Map-based growth of {builtup2 - builtup1:+.1f} points since the earlier "
                        "date; see the Overview tab for the noise caveat."))

    if expansion_mask is not None and hotspot_grid is not None:
        total_expansion_km2 = expansion_mask.sum() / CFG.PIXELS_PER_KM2
        top10_km2 = np.sort(hotspot_grid.flatten())[::-1][:10].sum() / CFG.PIXELS_PER_KM2
        diffuse_pct = (total_expansion_km2 - top10_km2) / total_expansion_km2 * 100 if total_expansion_km2 > 0 else 0
        status = "FLAG" if diffuse_pct >= 70 else "OK"
        indicators.append(("Development pattern", status, f"{diffuse_pct:.0f}% diffuse",
                            "Growth is spread broadly with a few concentrated hotspots -- "
                            "potential sprawl/fragmentation. Scattered changed pixels can also "
                            "come from classification noise, so confirm on imagery."
                            if status == "FLAG"
                            else "Development is relatively concentrated."))

    return indicators


# =================================================================
# Plot builders (return a Figure -- no file I/O)
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
              "Bare -> Built-up", "Other change (unverified; may include noise)"]

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
        st.error("Land-cover outputs not found in data/. Run notebooks/02_deployment_and_analysis.ipynb first.")
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
            m2.metric("Vegetated cover", f"{stats2['Green space']['pct']:.1f}%",
                      f"{stats2['Green space']['pct'] - stats1['Green space']['pct']:+.1f} pts")
            m3.metric("Cropland", f"{stats2['Cropland']['pct']:.1f}%",
                      f"{stats2['Cropland']['pct'] - stats1['Cropland']['pct']:+.1f} pts")

            gain, loss, net = built_up_flows(lc1, lc2)
            st.subheader("Built-up change (km^2)")
            f1, f2, f3 = st.columns(3)
            f1.metric("Net change", f"{net:+.2f}")
            f2.metric("Gross gain", f"{gain:.2f}", help="Non-built-up in 2021 -> built-up in 2026")
            f3.metric("Reverse flow", f"{loss:.2f}", help="Built-up in 2021 -> other class in 2026")
            st.caption("Real built-up loss over five years is expected to be small, so a large "
                       "reverse flow indicates classification noise between the two dates. When it "
                       "is comparable to the net change, the net growth is not statistically "
                       "established. Conversion shares and hotspots are derived from the gross "
                       "mask and share this noise.")

            st.subheader("Land-use balance, 2026")
            df = pd.DataFrame({
                "Class": list(stats2.keys()),
                "Area (km^2)": [round(v["km2"], 2) for v in stats2.values()],
                "Share (%)": [round(v["pct"], 1) for v in stats2.values()],
            })
            st.dataframe(df, width="stretch", hide_index=True)
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
        st.caption("Post-classification comparison: each date is classified independently, so an "
                   "error in either map appears as change.")
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
            st.caption("Hotspots are counted on the gross expansion mask, which includes "
                       "classification noise.")
            fig_hot = plot_hotspots(hotspot_grid, rgbnir2)
            st.pyplot(fig_hot)
            fig_download_button(fig_hot, "development_hotspots.png")
        else:
            st.info("Hotspot data not found -- run notebooks/02_deployment_and_analysis.ipynb first.")

    # --- Model Validation ---
    with tab_validation:
        st.subheader("Held-out test set performance")
        st.caption("Measured on 5 labeled rasters never seen during training (pooled pixel IoU, "
                   "flip test-time augmentation) -- not this study area specifically. The test "
                   f"set is small: only {CFG.TEST_BUILTUP_DOMINANT_PATCHES} of 13,407 test patches "
                   "are built-up-dominant, so built-up and cropland scores are statistically fragile.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean IoU", f"{CFG.TEST_MEAN_IOU:.3f}")
        c2.metric("FWIoU", f"{CFG.TEST_FWIOU:.3f}")
        c3.metric("Precision", f"{CFG.TEST_PRECISION:.3f}")
        c4.metric("Recall", f"{CFG.TEST_RECALL:.3f}")
        c5.metric("F1", f"{CFG.TEST_F1:.3f}")

        st.divider()

        st.subheader("Held-out-area check vs. ESA WorldCover 2021")
        st.caption("WorldCover 2021 is the source of the training labels and has its own errors, "
                   "so this measures consistency with a reference product, not accuracy against "
                   "ground truth. The study area lies outside every training raster. Note: "
                   "grid-resized for comparison rather than fully CRS-reprojected, so this is "
                   "approximate, not a survey-grade confusion matrix.")
        st.metric("Overall pixel agreement", f"{CFG.WORLDCOVER_OVERALL_AGREEMENT:.1f}%")

        wc_df = pd.DataFrame({
            "Class": list(CFG.WORLDCOVER_PER_CLASS.keys()),
            "IoU vs WorldCover (%)": list(CFG.WORLDCOVER_PER_CLASS.values()),
        })
        st.dataframe(wc_df, width="stretch", hide_index=True)

        if CFG.WORLDCOVER_FIG_PATH.exists():
            st.image(str(CFG.WORLDCOVER_FIG_PATH), width="stretch")
        else:
            st.info("WorldCover comparison figure (figures/worldcover_validation.png) not found.")

        st.divider()

        st.subheader("Water-class consistency check (NDWI)")
        st.caption("Computed live from the study-area imagery: pixels with NDWI > 0 are used as a "
                   "reference water mask. NDWI is a reference index, not ground truth (shadows, "
                   "including cloud shadow, can also have NDWI > 0). Precision = share of "
                   "model-water pixels that NDWI also marks as water; recall = share of NDWI-water "
                   "pixels the model also marks as water.")
        ndwi_cols = st.columns(2)
        for col, (date_label, img, lc_d) in zip(
                ndwi_cols, [("2021", rgbnir1, lc1), ("2026", rgbnir2, lc2)]):
            with col:
                if img is not None:
                    res = ndwi_water_check(img, lc_d)
                    st.markdown(f"**{date_label}** -- model water {res['model_km2']:.2f} km^2, "
                                f"NDWI water {res['ndwi_km2']:.2f} km^2")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Precision", f"{res['precision']:.2f}")
                    p2.metric("Recall", f"{res['recall']:.2f}")
                    p3.metric("IoU", f"{res['iou']:.2f}")
                else:
                    st.info(f"Imagery for {date_label} not found in data/.")

        st.divider()

        st.subheader("Known limitations")
        st.markdown(
            "- **Labels are not ground truth.** The model learns ESA WorldCover, a model-derived "
            "product with its own errors; no independent ground truth exists for this area.\n"
            "- **Bare/sparse vs. Low vegetation boundary confusion**: the WorldCover comparison "
            "above indicates the model over-predicts Bare/sparse where the reference shows Low "
            "vegetation. Bare/sparse <-> Built-up is also a likely confusion pair, which affects "
            "the conversion-source shares.\n"
            "- **Small test set**: its geographic tile composition underrepresents Built-up and "
            "Cropland relative to train/validation (4 of 5 test rasters contain almost no "
            "built-up), so the test scores for those two classes are statistically fragile "
            "rather than reliably higher or lower.\n"
            "- **Change detection by post-classification comparison**: errors in either map create "
            "apparent change, and the transition matrix is strongly asymmetric between some "
            "classes, which points to systematic differences between the two dates "
            "(vegetation greenness or radiometry), not only random noise.\n"
            "- **Imagery**: one June scene per year, so rainfall-driven vegetation differences "
            "cannot be separated from real change. No cloud/shadow mask (SCL) was applied; cloud "
            "shadow in the 2026 scene may affect water extent near the northern lake.\n"
            "- **Inference**: non-overlapping patches and no test-time augmentation, unlike the "
            "test evaluation, so patch seams are possible."
        )

    # --- Planning Indicators ---
    with tab_indicators:
        st.caption("Thresholds are illustrative rule-of-thumb values, not official planning "
                   "standards, and every indicator inherits the classification errors of the "
                   "underlying maps. Read FLAG as 'worth checking', not as a confirmed finding.")
        indicators = compute_planning_indicators(stats1, stats2, expansion_mask, hotspot_grid,
                                                 lc1, lc2)
        for name, status, value, message in indicators:
            if status == "FLAG":
                st.error(f"**{name}**: {value} -- {message}")
            elif status == "WATCH":
                st.warning(f"**{name}**: {value} -- {message}")
            else:
                st.success(f"**{name}**: {value} -- {message}")


if __name__ == "__main__":
    main()
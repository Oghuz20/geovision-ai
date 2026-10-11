# GeoVision AI

**AI-Based Urban Area Analysis and Planning Support System Using Satellite Imagery**

Submitted to the Azercosmos Earth Remote Sensing Competition 2026 (Urban Planning and
Architecture theme).

An AI-based system for automated land-cover classification and multi-temporal urban growth
monitoring using Sentinel-2 satellite imagery, applied to a real case study area in Azerbaijan
(Khırdalan–Masazır–Binagadi, Absheron peninsula).

## Live demo

**Dashboard:** [geovision-ai-baku.streamlit.app](https://geovision-ai-baku.streamlit.app/)

## Results summary

- **Segmentation model** (U-Net++/EfficientNet-B5 + U-Net/EfficientNet-B3 ensemble): **0.611
  mean IoU on a held-out test set** of 5 rasters (pooled pixel IoU, flip test-time augmentation).
  Per-class test IoU ranges from 0.97 (water) to 0.34 (cropland); built-up is 0.41. The test set
  is small and geographically narrow (see [Limitations](#limitations)), so treat these numbers as
  indicative.
- **Held-out-area check:** on the study area, the 2021 map agrees with ESA WorldCover 2021 on
  **71.0% of pixels**. Per-class IoU against WorldCover is reasonable for built-up (0.78) and
  water (0.66) but weak for low vegetation (0.37), bare/sparse (0.19), cropland (0.12) and tree
  cover (0.06). WorldCover is also the source of the training labels and has its own errors, so
  this is a *consistency check against a reference product*, not accuracy against ground truth.
  The study area lies outside every training raster (see [Dataset](#dataset)).
- **Water consistency (NDWI):** the model's water class agrees only weakly with an NDWI > 0
  reference mask (2021: precision 0.18, recall 0.59, IoU 0.16; 2026: precision 0.34, recall
  0.82, IoU 0.31). The model labels more water than NDWI does; the cause has not been isolated
  (see [Limitations](#limitations)).
- **Built-up change 2021–2026:** net **+5.26 km²** (88.40 → 93.66 km²; 55.75% → 59.06% of the
  study area, +3.3 percentage points). The gross non-built-up → built-up area is 11.67 km², but
  6.41 km² of built-up also flips to other classes, which is implausible for real change over five
  years. This suggests a classification-noise floor of the same order as the net change, so the
  net growth is **not established as statistically significant**.
- **Conversion sources** (of the 11.67 km² gross): 72% (8.41 km²) from bare/sparse land, 26%
  (2.99 km²) from low vegetation and about 1% (0.12 km²) directly from cropland. These shares come
  from the gross change mask, and bare/sparse is a known confusion class, so they are indicative
  only.
- **Cropland:** mapped cropland falls from 3.54 to 1.10 km² (−2.44 km²). Only 0.12 km² of that
  goes to built-up; 2.16 km² goes to low vegetation, a known confusion pair. With a cropland test
  IoU of 0.34 and a study-area IoU of 0.12 against WorldCover, this is a **monitoring signal, not
  confirmed loss**.
- **Largest map changes are not urban expansion:** between the two dates bare/sparse falls by
  17.0 km² (−65%) and water/wetland rises by 7.0 km² (+52%), both larger than the built-up
  change (+5.3 km²). This points to date-to-date differences (radiometry, season, cloud shadow)
  rather than only real land-use change.
- **Planning indicators:** four rule-based indicators with illustrative (non-official) thresholds.
  Flagged: agricultural land loss (monitoring signal) and diffuse development pattern (may partly
  reflect classification noise). Built-up saturation is on watch; vegetated cover is above its
  reference level.

## Dataset

- **117,142 patches**, 256×256 px, four bands (R, G, B, NIR), cut with **stride 128 (~50%
  overlap)**, so neighbouring patches are strongly correlated and the effective sample size is much
  smaller than the patch count.
- **30 rasters**, each a 12,000×12,000 px (1°×1°, 1/12,000° grid) export from the south-west
  corner of a different ESA WorldCover 3° grid cell, spread across the globe. Imagery was exported
  with Google Earth Engine; labels are ESA WorldCover 2021 (v200) remapped to six classes plus an
  ignore class.
- **Split by raster** (no raster appears in two splits): 21 train / 4 val / 5 test rasters
  (90,221 / 13,514 / 13,407 patches). Images are stored as uint8 (reflectance 0–10,000 clipped and
  scaled to 0–255).
- **Class balance:** built-up is only ~2.2% of pixels on average.
- **Study-area independence:** each training raster lies inside its own WorldCover cell. The study
  area (49.73–49.87°E, 40.41–40.53°N) falls in cell `N39E048`, whose exported raster covers only
  48–49°E, 39–40°N (about 75 km away), so no training pixel overlaps the study area.

## Repository structure

```
geovision-ai/
├── README.md
├── requirements.txt                # dashboard only (what Streamlit Cloud installs)
├── requirements-pipeline.txt       # training / study-area scripts (PyTorch, rasterio, ...)
├── app.py                          # Streamlit decision-support dashboard
│
├── notebooks/
│   ├── 01_training_pipeline.ipynb          # Model training (GPU required)
│   └── 02_deployment_and_analysis.ipynb    # AOI inference, change detection,
│                                            # hotspots, indicators, figures
│
├── src/                            # The notebook-02 pipeline as runnable scripts
│   ├── phase4_segment_aoi.py               # ensemble segmentation of both dates
│   ├── phase5_6.py                         # area statistics + change detection
│   ├── phase7_hotspot.py                   # 500 m hotspot grid
│   ├── phase8_planning_indicators.py       # rule-based indicators
│   ├── phase9_figures.py                   # figures 1-8 (10 PNG files)
│   └── worldcover_validation.py            # WorldCover + NDWI consistency checks
│
├── checkpoints/
│   └── README.md                   # Trained model download link (not stored in-repo)
│
├── data/
│   ├── aoi.geojson
│   ├── 2021_RGBNIR_aoi.npy         # AOI-cropped Sentinel-2 imagery
│   ├── 2026_RGBNIR_aoi.npy
│   ├── 2021_landcover.npy          # Model output land-cover maps
│   ├── 2026_landcover.npy
│   ├── nonbuiltup_to_builtup_mask.npy
│   ├── hotspot_grid.npy
│   └── transition_matrix.csv
│
└── figures/
    ├── fig1_study_area.png, fig2_segmentation.png
    ├── fig3_builtup_2021.png, fig3_builtup_2026.png
    ├── fig4_vegetation_2021.png, fig4_vegetation_2026.png
    ├── fig5_comparison.png, fig6_change_detection.png
    ├── fig7_expansion_map.png, fig8_hotspots.png
    └── worldcover_validation.png
```

## Reproducing the results

1. **Training data:** the full training dataset (117,142 labeled Sentinel-2 patches, sharded
   as `.npz`) is hosted on Kaggle:
   [geovision-sentinel2-shards](https://www.kaggle.com/datasets/oghuzhasanli/geovision-sentinel2-shards).
2. **Train the model** — open `notebooks/01_training_pipeline.ipynb` on a GPU environment
   (Kaggle/Colab). Run once with `CFG.MODEL_NAME="Unet"` / `CFG.ENCODER_NAME="efficientnet-b3"`,
   and once with `CFG.MODEL_NAME="UnetPlusPlus"` / `CFG.ENCODER_NAME="efficientnet-b5"`, to
   produce both ensemble members.
3. **Checkpoints:** download the pre-trained checkpoints from the link in `checkpoints/README.md`
   if you'd rather skip step 2.
4. **Run the case study** — either open `notebooks/02_deployment_and_analysis.ipynb` or run the
   equivalent scripts from the repository root, in this order (CPU is enough):
   ```bash
   pip install -r requirements-pipeline.txt
   python src/phase4_segment_aoi.py        # needs the checkpoints from step 3
   python src/phase5_6.py
   python src/phase7_hotspot.py
   python src/phase8_planning_indicators.py
   python src/phase9_figures.py
   python src/worldcover_validation.py     # needs data/worldcover2021_aoi_remapped.npy
   ```
   This reproduces the study-area results, the figures and the WorldCover/NDWI checks. The
   test-set metrics shown in the dashboard come from the evaluation in notebook 01.
5. **Launch the dashboard:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Data sources

- **Training imagery:** Sentinel-2 RGB+NIR exported through Google Earth Engine.
- **Study-area imagery:** [Sentinel-2 L2A](https://dataspace.copernicus.eu/), Copernicus Data
  Space Ecosystem (ESA/EU open data license), tile T39TUE, June 2021 and June 2026. The 2026
  product needed the `BOA_ADD_OFFSET` radiometric correction (processing baseline 04.00, in force
  since 25 January 2022).
- **Training labels and reference for the held-out-area check:**
  [ESA WorldCover 2021 (v200)](https://esa-worldcover.org/).

## Limitations

- **Labels are not human ground truth.** Training labels are ESA WorldCover, a model-derived
  product with its own errors; the model learns to reproduce it. No independent ground truth
  exists for the study area.
- **Small, unbalanced test set.** Only 44 of 13,407 test patches are built-up-dominant; four of
  the five test rasters contain under 0.2% built-up, so the built-up test IoU (0.41) rests mainly
  on one raster. Cropland-dominant patches are 6.9% of test versus ~27% of train and validation.
- **Validation and test metrics are not comparable.** Validation mIoU (used for checkpoint
  selection) is the mean of per-batch IoUs; test mIoU is pooled over all pixels. The gap between
  e.g. 0.414 (val) and 0.611 (test) is therefore not a generalization gap.
- **Change detection is post-classification comparison.** Errors in either date's map create
  apparent change. Transitions are strongly asymmetric (for example built-up → tree cover far
  exceeds tree cover → built-up), which points to systematic differences between the two dates
  (vegetation greenness or radiometry) rather than random noise, so even the net change is
  affected. Conversion shares, hotspots and the "diffuse development" indicator all use the gross
  expansion mask and share this uncertainty.
- **Water class vs NDWI.** On the study area the model labels about 3× more water than an
  NDWI > 0 mask in 2021 (precision 0.18) and about 2.4× in 2026 (precision 0.34). NDWI computed on
  uint8 bands is itself a crude reference, and shadows, dark surfaces, salt or oil lakes and the
  wetland definition could each contribute; the cause has not been isolated.
- **Date-to-date class swings.** Mapped bare/sparse falls by 17.0 km² (−65%) and water/wetland
  rises by 7.0 km² (+52%) between 2021 and 2026, both larger than the built-up change
  (+5.3 km²). These likely reflect radiometric or seasonal differences between the two scenes, so
  conversion shares and hotspots should not be read as confirmed land-use change.
- **Edge effects.** Inference pads the study area with zeros to a multiple of 256 px; the last
  patch column is 29% padding and the last row 86%. Five of the ten strongest hotspot cells lie
  within about 1 km of the right or bottom edge of the study area and may be affected. The
  hotspots in the interior (around rows 450–600, columns 150–300 of the study-area array) are the
  better candidates for real development. None of the hotspots has yet been confirmed on
  high-resolution imagery.
- **Imagery.** One June scene per year, so rainfall-driven vegetation differences cannot be
  separated from real change. No cloud or cloud-shadow mask (Sentinel-2 SCL) was applied over the
  study area; the reported ~10% cloud cover is scene-level, and the 2026 scene shows cloud shadow
  near the northern lake.
- **Training/deployment domain shift.** Training imagery was exported via Google Earth Engine on
  a 1/12,000° geographic grid; the study-area imagery is native 10 m UTM from Copernicus with a
  separate BOA-offset correction.
- **Inference.** Study-area maps use non-overlapping 256×256 patches without test-time
  augmentation (the 0.611 test figure used flip-TTA), so patch seams are possible.
- **Indicators.** Thresholds (15% vegetated cover, 25% cropland loss, 55/60% built-up
  saturation, 70% diffuse) are illustrative, not official planning standards. "Vegetated cover" is
  tree plus low vegetation and includes semi-desert scrub, not only urban green space.
- **Quantization.** Images are stored as uint8, which reduces spectral precision, especially for
  dark surfaces such as water.

## Planned improvements

- Remove small connected components (minimum mapping unit) and require probability agreement
  between dates before counting a pixel as changed.
- Apply a Sentinel-2 SCL cloud/shadow mask and use multi-date seasonal composites.
- Sliding-window inference with stride 128, reflect padding and test-time augmentation; check
  whether the edge hotspots persist.
- Confirm the strongest hotspots visually on high-resolution imagery.
- Estimate change area with confidence intervals from a stratified sample of reference points
  interpreted on high-resolution imagery.

## License

Code in this repository is provided for the purposes of the Azercosmos Earth Remote Sensing
Competition 2026. Satellite imagery used is subject to the Copernicus open data license; ESA
WorldCover data is subject to its own CC BY 4.0 license.
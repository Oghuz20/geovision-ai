# GeoVision AI

**AI-Based Urban Area Analysis and Planning Support System Using Satellite Imagery**

Submitted to the Azərkosmos Earth Remote Sensing Competition 2026 (Urban Planning and
Architecture theme).

An AI-based system for automated land-cover classification and multi-temporal urban growth
monitoring using Sentinel-2 satellite imagery, applied to a real case study area in Azerbaijan
(Khırdalan–Masazır–Binagadi, Absheron peninsula).

## Live demo

**Dashboard:** [geovision-ai-baku.streamlit.app](https://geovision-ai-baku.streamlit.app/)

## Results summary

- Segmentation model (U-Net++/EfficientNet-B5 + U-Net/EfficientNet-B3 ensemble): **0.611 test
  mean IoU**, independently validated against ESA WorldCover 2021 ground truth (**71.0%
  agreement**) for the study area.
- Built-up area grew **55.75% → 59.06%** of the study area (**+5.9%**, 2021–2026).
- **72%** of new built-up area developed on previously bare/vacant land; only 1% through direct
  cropland conversion.
- Two flagged planning indicators: agricultural land loss and diffuse development pattern
  (potential sprawl).

## Repository structure

```
geovision-ai/
├── README.md
├── requirements.txt
├── app.py                          # Streamlit decision-support dashboard
│
├── notebooks/
│   ├── 01_training_pipeline.ipynb          # Model training (GPU required)
│   └── 02_deployment_and_analysis.ipynb    # AOI inference, change detection,
│                                            # hotspots, indicators, figures (CPU-only)
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
    ├── fig1_study_area.png ... fig8_hotspots.png
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
4. **Run the case study** — `notebooks/02_deployment_and_analysis.ipynb` (CPU-only) takes the
   trained checkpoints and the AOI imagery in `data/`, and reproduces every result and figure.
5. **Launch the dashboard:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Data sources

- **Satellite imagery:** [Sentinel-2 L2A](https://dataspace.copernicus.eu/), Copernicus Data
  Space Ecosystem (ESA/EU open data license), tile T39TUE, June 2021 and June 2026.
- **Training labels & independent validation:** [ESA WorldCover 2021 (v200)](https://esa-worldcover.org/).

## License

Code in this repository is provided for the purposes of the Azərkosmos Earth Remote Sensing
Competition 2026. Satellite imagery used is subject to the Copernicus open data license; ESA
WorldCover data is subject to its own CC BY 4.0 license.

# Trained model checkpoints

The two checkpoints that make up the final ensemble are not stored in this repository —
GitHub blocks files over 100MB, and both checkpoints exceed that individually.

| File | Architecture | Encoder | Val mIoU | Size |
|---|---|---|---|---|
| `unet_best_model.pth` | U-Net | EfficientNet-B3 | 0.4143 | 146 MB |
| `unetpp_best_model.pth` | U-Net++ | EfficientNet-B5 | 0.4330 | 358 MB |

**Download link:** [Google Drive folder](https://drive.google.com/drive/folders/1En3JR3uFnyLIl-3hyJOgK7cxuSxusp4P?usp=sharing)

Place both files in this `checkpoints/` folder before running
`notebooks/02_deployment_and_analysis.ipynb`, or update the paths in
`CFG.ENSEMBLE_CONFIGS` (in the notebook) to point wherever you've saved them.
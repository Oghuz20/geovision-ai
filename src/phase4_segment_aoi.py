"""Phase 4 -- segment the study area for both dates with the two-model ensemble.

Run from the repo root:  python phase4_segment_aoi.py
Reads : data/2021_RGBNIR_aoi.npy, data/2026_RGBNIR_aoi.npy, checkpoints (see CFG)
Writes: data/2021_landcover.npy, data/2026_landcover.npy
"""
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp


class CFG:
    DATA_DIR = Path("data")

    DATE1_LABEL, DATE2_LABEL = "2021", "2026"
    DATE1_AOI_PATH = DATA_DIR / "2021_RGBNIR_aoi.npy"
    DATE2_AOI_PATH = DATA_DIR / "2026_RGBNIR_aoi.npy"
    DATE1_LC_PATH = DATA_DIR / "2021_landcover.npy"
    DATE2_LC_PATH = DATA_DIR / "2026_landcover.npy"

    # Update to your own checkpoint locations -- see checkpoints/README.md
    ENSEMBLE_CONFIGS = [
        {"path": "checkpoints/unetpp_best_model.pth", "model_name": "UnetPlusPlus", "encoder": "efficientnet-b5"},
        {"path": "checkpoints/unet_best_model.pth",   "model_name": "Unet",         "encoder": "efficientnet-b3"},
    ]

    IN_CHANNELS = 5
    NUM_CLASSES = 7
    PATCH_SIZE = 256
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_ndvi_channel(img_uint8):
    img = img_uint8.astype(np.float32)
    red, nir = img[..., 0], img[..., 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi_scaled = ((ndvi + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return np.dstack([img_uint8, ndvi_scaled])


def pad_to_multiple(arr, patch_size=CFG.PATCH_SIZE):
    h, w = arr.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    return padded, (h, w)


def patchify(padded_arr, patch_size=CFG.PATCH_SIZE):
    h, w = padded_arr.shape[:2]
    patches, coords = [], []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patches.append(padded_arr[y:y + patch_size, x:x + patch_size])
            coords.append((y, x))
    return patches, coords


def load_ensemble_model(cfg, device):
    ModelClass = getattr(smp, cfg["model_name"])
    model = ModelClass(
        encoder_name=cfg["encoder"], encoder_weights=None,
        in_channels=CFG.IN_CHANNELS, classes=CFG.NUM_CLASSES, activation=None,
    ).to(device)
    ckpt = torch.load(cfg["path"], map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {cfg['model_name']}/{cfg['encoder']} from {cfg['path']}")
    return model


def run_inference_ensemble(models, patches, coords, padded_shape, device, batch_size=CFG.BATCH_SIZE):
    full_pred = np.zeros(padded_shape[:2], dtype=np.uint8)
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            batch_coords = coords[i:i + batch_size]
            batch_arr = np.stack([add_ndvi_channel(p) for p in batch_patches]).astype(np.float32) / 255.0
            batch_tensor = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2)).to(device)

            probs_sum = None
            for m in models:
                probs = torch.softmax(m(batch_tensor), dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            preds = torch.argmax(probs_sum / len(models), dim=1).cpu().numpy()

            for pred, (y, x) in zip(preds, batch_coords):
                full_pred[y:y + CFG.PATCH_SIZE, x:x + CFG.PATCH_SIZE] = pred
    return full_pred


def segment_full_area(models, rgbnir_uint8, device):
    padded, orig_shape = pad_to_multiple(rgbnir_uint8)
    patches, coords = patchify(padded)
    print(f"  {len(patches)} patches ({padded.shape[:2]} padded from {orig_shape})")
    full_pred_padded = run_inference_ensemble(models, patches, coords, padded.shape, device)
    return full_pred_padded[:orig_shape[0], :orig_shape[1]]


def main():
    print("Device:", CFG.DEVICE)
    models = [load_ensemble_model(cfg, CFG.DEVICE) for cfg in CFG.ENSEMBLE_CONFIGS]

    date1_crop = np.load(CFG.DATE1_AOI_PATH)
    date2_crop = np.load(CFG.DATE2_AOI_PATH)
    print(f"{CFG.DATE1_LABEL} crop shape:", date1_crop.shape)
    print(f"{CFG.DATE2_LABEL} crop shape:", date2_crop.shape)

    print(f"\nSegmenting {CFG.DATE1_LABEL}...")
    lc1 = segment_full_area(models, date1_crop, CFG.DEVICE)
    print(f"Segmenting {CFG.DATE2_LABEL}...")
    lc2 = segment_full_area(models, date2_crop, CFG.DEVICE)

    np.save(CFG.DATE1_LC_PATH, lc1)
    np.save(CFG.DATE2_LC_PATH, lc2)
    print("Saved land-cover maps.")

    # Sanity check: class 0 is the "ignore" channel and should (almost) never be predicted.
    for label, lc in [(CFG.DATE1_LABEL, lc1), (CFG.DATE2_LABEL, lc2)]:
        print(f"{label}: share of pixels predicted as class 0 (ignore) = {(lc == 0).mean():.6f}")


if __name__ == "__main__":
    main()
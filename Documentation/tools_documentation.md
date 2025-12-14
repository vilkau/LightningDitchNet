# `utils/tools.py` — Terrain Feature & Hyperparameter Utilities

`tools.py` provides utility functions for feature generation, normalization, 
and hyperparameter loading. The functions produce the terrain-based HPMF and ISI layers used by the model, 
performs min–max scaling, and retrieves relevant model hyperparameters for testing and inference from `.yaml` files.

---

### `minmax_normalized_image(image, constant_fill_value)`
Normalizes an input raster to the range **[0, 1]** while handling no‑data values robustly. The function:
- Detects NaN values and common no‑data placeholders (`-9999`, `-32768`).
- Returns a constant array if the image has no variation or contains only invalid pixels.
- Scales valid pixels using `MinMaxScaler`.
- Fills invalid pixels with the mean of the scaled values to avoid holes in the result.
- Returns a float32 NumPy array with no‑data handled safely.

---

### `create_hpmf_layer(dem_path, hpmf_path)`
Generates the **High‑Pass Median Filter (HPMF)** layer from a DEM using WhiteboxTools.
- Applies an 11×11 median filter and subtracts the low‑frequency component to highlight local depressions.
- Returns the output raster as a NumPy array.

---

### `create_isi_layer(dem_path, isi_path)`
Computes the **Impoundment Size Index (ISI)** from the input DEM using WhiteboxTools.
- Measures the upstream impoundment size (in pixels) using a dam length of 6.
- Reads and returns the output raster as a NumPy array.

---

### `create_feature_layer(hpmf_array, isi_array, resampled_height, resampled_width)`
Combines the HPMF and ISI rasters into a **2‑channel feature tensor** for model input. The function:
- Stacks the two layers into shape [2, H, W].
- Resizes both channels to the target resolution.
- Returns a resized two‑channel float32 NumPy array ready for inference.

### `fetch_hparams_from_yaml(mode: Literal["test", "inference"], yaml_path)`

- Loads required model hyperparameters from a `.yaml` file saved during training.
- Extracts the following essential hyperparameters from `.yaml` file:
  - `encoder_name`
  - `in_channels`
  - `pos_weight`
- In inference mode, only returns what is needed:
  - `encoder_name` 
  - `in_channels`

This ensures consistent reconstruction of model configuration during evaluation or inference

## Dependencies
- **NumPy**: array manipulation, masking, stacking, and numerical operations.
- **scikit-image**, **tifffile**: for reading and resizing TIFF images.
- **scikit-learn**: for performing min–max normalization on valid raster pixels.
- **WhiteboxTools**: computes the HPMF and ISI terrain analysis layers used as model inputs.

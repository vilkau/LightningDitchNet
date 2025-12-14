# `inference.py` — Inference Pipeline for LightningDitchNet

## Overview
`inference.py` performs inference on DEM tiles using one or more trained `LightningDitchNet` model checkpoints.
During inference each model produces a prediction for every tile, and the outputs are averaged to create the final result.

---

## Class: `Inference`
Encapsulates all end-to-end inference logic, including model loading, feature generation, tiling, prediction, and output formatting.

### Initialization
The class is most commonly initialized through the CLI (`Main` class) or via the `DEM2Ditch` application, 
and its underlying structure is defined using the `InferenceConfig` from `config.py` as follows:

```python
Inference(config: InferenceConfig)
```

All configuration attributes are documented in the `Main` section, 
where each parameter is explained in a table.

### Main Responsibilities
- Loads all model checkpoints and corresponding hyperparameters from the model directory.
- Moves each model to the selected device (CPU/GPU).
- Generates the required feature layers (HPMF, ISI).
- Performs inference on 512×512 feature chips.
- Averages predictions from all models.
- Writes probability, binary, and depth rasters.
- Creates VRT mosaics for each enabled output type.
- Removes temporary files created during processing

---

### Methods

#### `_init_device()`
Determines the computation device:
- If `device="auto"`, selects `cuda` when available, otherwise `cpu`.
- Otherwise, returns the user-specified device.

#### `_init_models()`
Initializes all selected model checkpoints:
- Ensures that the directory contains at least one `.ckpt` file
- For each checkpoint:
  - Loads necessary model hyperparameters (`encoder_name`, `in_channels`) using `fetch_hparams_from_yaml(mode="inference")`
  - Constructs a `ModelConfig`
  - Initializes a new `LightningDitchNet` model
  - Moves the model to the selected device
  - Sets evaluation mode
- Stores all models for later ensemble prediction

This enables **multi-model averaging** by default.

<div style="border: 1.5px solid #d3d3d3; border-radius: 6px; padding: 10px;">

⚠️ **Note on Model Directory Requirements** ⚠️

The model directory must contain, in addition to the checkpoint files, a matching `.yaml` hyperparameter file for each checkpoint.

Each `.yaml` file must share the exact same filename prefix as its corresponding `.ckpt` file (e.g., `model_01.ckpt` → `model_01.yaml`).

These `.yaml` files are required for reconstructing the model configuration during inference.

</div>

#### `_set_output_directories()` and `_set_temporary_directories()`:
Creates a standardized directory layout for inference results. 
Only the subdirectories corresponding to the enabled output types are created:

- `probability_maps/` — continuous probability rasters
- `binary_maps/` — binary rasters
- `depth_maps/` — HPMF-based depth rasters
- `temp/` — intermediate files (removed automatically after processing)

#### `_create_output_layer()`
Runs tile-based multi-model inference:
- Splits the 2-channel feature raster into overlapping 512×512 tiles.
- Converts each tile to a PyTorch tensor.
- For each model:
  - Performs the model’s forward pass with gradient computation disabled.
  - Applies sigmoid activation function.
- Averages predictions across all models.
- Writes each tile back into the correct position in the output mosaic.
- Resamples the prediction raster back to the original DEM resolution.

#### `_output_probability_map(input_path, profile, output_array)`
- Saves the model’s continuous probability predictions as a float32 GeoTIFF.
- Updates the raster profile accordingly and writes the single-band probability map 
to the `probability_maps/` directory using the input DEM’s name as the filename prefix.

#### `_output_binary_map(input_path, profile, output_array)`
- Creates a binary ditch/no-ditch map from the probability raster using 
the user-defined threshold (self.threshold).
- Updates the raster profile accordingly and writes the single-band binary map 
to the `binary_maps/` directory using the input DEM’s name as the filename prefix.

#### `_output_depth_map(input_path, output_array)`
- Loads the corresponding HPMF reference and replaces no-data values with zero.
- Builds the ditch depth map by combining the probability output with the temporary HPMF raster.
- Removes all positive values to preserve only negative (depression) depths.
- Saves the resulting depth raster into the `depth_maps/` directory using the input DEM’s name 
as the filename prefix.

#### `_process_single_dem(dem_path)`
This method performs all steps required to transform one DEM input into its final probability, 
binary, and/or depth outputs:
- Opens the DEM and extracts spatial metadata and dimensions.
- Skips processing if the raster is smaller than 500×500 pixels.
- Generates the HPMF and ISI feature layers using utility functions and stores them in temporary files.
- Normalizes both features and applies a **2.4% upscale** to match the model’s expected resolution.
- Stacks the two normalized feature layers into a 2-channel array.
- Calls `_create_output_layer()` to perform tile-based model inference.
- Writes the selected output maps by invoking:
  - `_output_probability_map()`
  - `_output_binary_map()` (if enabled)
  - `_output_depth_map()` (if enabled)
- Removes intermediate temporary files tied to the processed DEM.

#### `_create_virtual_rasters()`
Creates VRT mosaic files for all output types that were enabled.
For each map category (probability, binary, depth), the method:

- Collects all `.tif` files in the corresponding output directory.
- Builds a GDAL VRT mosaic that references these rasters without duplicating data.
- Saves the VRT using a standardized filename:
  - `ditch_probability_map.vrt`
  - `ditch_binary_map.vrt`
  - `ditch_hpmf_depth_map.vrt`

These mosaics allow all tile-based outputs to be viewed as seamless layers in GIS software.

#### `predict()`
Serves as the main entry point for running inference across an entire directory of DEM files.
The method:
- Scans the input directory and collects all `.tif` DEM files.
- Prints a processing summary and device information (CPU or GPU).
- Iterates through each DEM and executes `_process_single_dem()`.
- Keeps track of how many DEMs were processed, skipped, or failed.
- After all DEMs are handled, creates VRT mosaics using `_create_virtual_rasters()` for the enabled output types.
- Cleans up the temporary working directory.
- Prints a final summary listing processed, skipped, and failed files.

### Output

After running inference, the script produces the following directory structure inside the specified output folder:

```
output_dir/
├── probability_maps/ (if enabled)
│   ├── dem_tile_001_ditch_probability.tif
│   ├── dem_tile_002_ditch_probability.tif
│   └── ...
│
├── binary_maps/ (if enabled)
│   ├── dem_tile_001_ditch_binary.tif
│   ├── dem_tile_002_ditch_binary.tif
│   └── ...
│
├── depth_maps/ (if enabled)
│   ├── dem_tile_001_ditch_depth.tif
│   ├── dem_tile_002_ditch_depth.tif
│   └── ...
│
└── temp/ (removed automatically after processing)
    ├── hpmf_temp.tif
    └── isi_temp.tif
```

---

## Class: `Main`
Provides the command-line interface for running inference.

### Arguments
**Inference Arguments** (via `add_inference_args` in `inference_args.py`)

| Argument          | Type  | Default | Description                                                                                                                                          |
|-------------------|-------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_dir`       | Path  | —       | Directory containing **one or more** trained `LightningDitchNet model` checkpoints (`.ckpt`) <br/> and corresponding hyperparameter files (`.yaml`). |
| `input_dem_dir`   | Path  | —       | Directory containing the DEM tiles (`.tif`) to be processed.                                                                                         |
| `output_dir`      | Path  | —       | Directory where all inference results will be written.                                                                                               |
| `--threshold`     | float | `0.3`   | Probability threshold used to generate the binary map.                                                                                               |
| `--no_prob_map`   | flag  | enabled | Disables saving of the probability map output.                                                                                                       |
| `--no_binary_map` | flag  | enabled | Disables saving of the binary map output.                                                                                                            |
| `--no_depth_map`  | flag  | enabled | Disables saving of the depth map output.                                                                                                             |
| `--device`        | str   | `auto`  | Specifies the computation device (`"cpu"` / `"cuda"` or `"auto"`). <br/> `"auto"` selects GPU when available, otherwise CPU.                         |

---

## Example Usage
```bash
python inference.py   ./models/   ./input_DEMs   ./inference_output --threshold 0.1
```

Both **relative** and **absolute** paths are supported for all input and output arguments.

### Example Output (Console)
```
Using device: cuda

The following models will be used and their predictions will be averaged:
model_01.ckpt
model_02.ckpt

Probability map output: enabled
Binary map output: enabled (threshold: 0.3)
Depth map output: enabled

Running DitchNet inference on DEM files in: ./input_DEMs

Processing: dem_tile_001.tif
Processing: dem_tile_002.tif
Processing: dem_tile_003.tif
...

All predictions completed.
```

---

## Dependencies
- **GDAL**: builds VRT mosaics that merge tile-based outputs into seamless virtual rasters.
- **model.py**: defines the `LightningDitchNet` architecture and enables loading the trained model checkpoint.
- **NumPy**: supports array manipulation for feature stacking, masking, and output raster creation.
- **PyTorch**: runs the trained DitchNet model and handles tensor operations on CPU or GPU.
- **Rasterio**: used for reading input DEMs, writing GeoTIFF outputs, and managing spatial metadata.
- **scikit-image**: provides the resize function used for resampling the prediction layer.
- **utils.py**: generates terrain feature layers (HPMF and ISI) and performs normalization required for inference.

The inference results serve as the final product of the `LightningDitchNet` workflow and can be directly visualized 
or used for GIS-based ditch mapping and analysis.

# `preprocessing.py` — Data Preprocessing Pipeline for LightningDitchNet

## Overview
`preprocessing.py` handles all **data preprocessing and chip generation** steps required for training and testing the `LightningDitchNet` segmentation model.  
It converts large digital elevation model (DEM) rasters and vector ditch data into small, standardized feature–label pairs (chips) suitable for model input.

---

## Class: `DitchDataset`
A PyTorch Dataset class used in both `train.py` and `test.py` scripts, 
providing feature and label tensors to the model during training, validation, or testing.

### Attributes
- **X** (`list[Path]`): Paths to input feature TIFFs.
- **y** (`list[Path]`): Paths to label TIFFs.
- **transform** (`albumentations.Compose`): Albumentations pipeline applied jointly to image and mask.

### Behavior
Each sample pair is:
- Loaded as a NumPy array (features as `float32`, labels as `uint8`).
- Channels of feature image are rearranged from `[C, H, W] → [H, W, C]`.
- The transform is applied consistently to both image and label.
- The label is binarized (`0`/`1`), cast to `float32`, and reshaped to `[1, H, W]`.

This ensures compatibility with the segmentation model’s input shape.

---

## Class: `ChipGenerator`
Responsible for **DEM processing, feature extraction, label creation, and chip generation**.

### Initialization
The class is most commonly initialized through the CLI (`Main` class) or via the `DEM2Ditch` application, 
and its underlying structure is defined using the `PreprocessingConfig` from `config.py` as follows:

```python
ChipGenerator(config: PreprocessingConfig)
```

All configuration attributes are documented in the `Main` section, 
where each parameter is explained in a table.

### Main Responsibilities
- Establishes directory hierarchy for `training_data` or `test_data`.
- Creates HPMF and ISI feature rasters using `WhiteboxTools`.
- Rasterizes and filters ditch vector geometries to form binary label rasters.
- Normalizes all layers and tiles them into `512×512` chips.
- Removes temporary rasters after processing.

### Methods

#### `_set_directories()`
Creates a standardized directory layout under the output folder, containing:
- `training_data/` or `test_data/` — main output directory depending on mode
- `feature_chips/` — extracted and normalized feature rasters
- `label_chips/` — generated label rasters
- `temp/` — intermediate files (removed automatically after processing)

#### `_create_label_layer(dem_path, hpmf_array, resampled_height, resampled_width)`
- Reads the corresponding DEM’s spatial metadata.
- Clips vector ditch geometries to DEM extent and buffers them by 1.5 m.
- Rasterizes buffered geometries and filters pixels using the HPMF threshold (`≤ -0.075` by default).
- Applies a 3×3 majority filter for smoothing.
- Resamples to align with model resolution by using nearest-neighbor interpolation.

<div style="border: 1.5px solid #d3d3d3; border-radius: 6px; padding: 10px;">

⚠️ **Coordinate System and Spatial Coverage Requirements** ⚠️

The input label vector data must share the same coordinate reference system (CRS) as the DEM data and fully cover the same spatial extent.  

If the coordinate systems differ or the vector layer does not overlap the DEM tile completely, the label generation and clipping process will fail.

</div>

#### `_generate_single_chip_pair(...)`
- Extracts matching `feature_chip` and `label_chip` arrays.
- Skips chips with < 0.1 % ditch pixels to reduce class imbalance.
- Saves both as `.tif` files (float32 and uint8 respectively).

#### `generate_chips()`
The main method controlling the full preprocessing pipeline:
- Iterates through all DEM files in the input directory.
- Generates temporary HPMF, ISI, and label rasters.
- Combines them into normalized two-channel feature arrays.
- Iteratively tiles the data into `512×512` chips.
- Removes all temporary files after completion.
- Reports skipped or invalid DEM inputs.

---

## Class: `Main`
Provides the command-line interface for running preprocessing.

### Arguments
**Preprocessing Arguments** (via `add_preprocessing_args` in `preprocesing_args.py`)

| Argument                 | Type  | Default  | Description                                                         |
|--------------------------|-------|----------|---------------------------------------------------------------------|
| `input_dem_dir`          | Path  | —        | Directory containing the DEM tiles `(.tif)` to be processed.        |
| `label_vector_data`      | Path  | —        | Vector dataset (e.g., `.shp`, `.gpkg`) containing ditch features.   |
| `output_dir`             | Path  | —        | Output directory for generated chips.                               |
| `--mode`                 | str   | `train`  | Determines subdirectory naming (`"train"` or `"test"`).             |
| `--ditch_width`          | float | `1.5`    | Buffer size (meters) applied to ditch vectors                       |
| `--label_hpmf_threshold` | float | `-0.075` | Threshold for ditch pixel selection in the HPMF layer (`≤ value`).  |

---

## Output
Depending on the mode, preprocessing produces the following structure:

```
output_dir/
└── training_data/ or test_data/
    ├── feature_chips/
    │   ├── 0.tif
    │   ├── 1.tif
    │   └── ...
    ├── label_chips/
    │   ├── 0.tif
    │   ├── 1.tif
    │   └── ...
    └── temp/ (removed automatically after processing)
        ├── hpmf_temp.tif
        ├── isi_temp.tif
        └── label_temp.tif
```

Each feature chip is a 2-channel raster (HPMF + ISI) normalized to `[0, 1]`,  
and each label chip is a binary raster (`1` = ditch, `0` = background).

---

## Example Usage
```bash
python preprocessing.py   ./input_DEMs   ./ditch_vectors/ditches.gpkg   ./dataset_output   --label_hpmf_threshold -0.05
```

Both **relative** and **absolute** paths are supported for all input and output arguments.  
This means you can run the program from any working directory without adjusting its internal path handling.

### Example Output (Console)
```
Running DitchNet preprocessing on DEM files in: ./input_DEMs

Mode: train
Ditch label width: 1.5
Label HPMF threshold: -0.05

Processing: dem_tile_001.tif
Processing: dem_tile_002.tif
...

Preprocessing completed.
```

---

## Dependencies
- **Albumentations**: for image augmentation and preprocessing in `DitchDataset`.
- **Rasterio**, **GeoPandas**, **Shapely**: handle raster and vector geospatial data.
- **scikit-image**, **tifffile**: for reading, writing, and resizing TIFF images.
- **utils.py**: provides helper functions for normalization and layer creation.
- **WhiteboxTools**: provides the `majority_filter` operation used for smoothing the label raster.

The output dataset integrates directly with `train.py` and `test.py` for model development.

# `test.py` — Evaluation Pipeline for LightningDitchNet

## Overview
`test.py` handles the **evaluation process** for the `LightningDitchNet` segmentation model. It loads a trained model 
checkpoint from `train.py`, prepares the test dataset using the same preprocessing and dataset pipeline, 
and computes validation metrics.

---

## Class: `Test`
Encapsulates all functionality required to evaluate the trained `LightningDitchNet` model.

### Initialization
The class is most commonly initialized through the CLI (`Main` class) or via the `DEM2Ditch` application, 
and its underlying structure is defined using the `TestConfig` from `config.py` as follows:

```python
Test(config: TestConfig)
```
All configuration attributes are documented in the `Main` section, 
where each parameter is explained in a table.

### Main Responsibilities
- Loads the test dataset using `DitchDataset` to ensure consistent preprocessing.
- Loads a trained `LightningDitchNet` checkpoint using the configuration paths.
- Builds a DataLoader for evaluation.
- Runs evaluation through a PyTorch Lightning Trainer.
- Logs evaluation metrics and writes results to `lightning_logs/test_logs/`.

---

### Methods

#### `_init_model()`
Loads the trained `LightningDitchNet` model using the configuration paths:
- Reads the model hyperparameters (`encoder_name`, `in_channels`, `pos_weight`) from the `hparams.yaml` 
file using `fetch_hparams_from_yaml(mode="test")`.
- Creates a `ModelConfig` instance populated with these values.
- Initializes a new `LightningDitchNet` model using the reconstructed configuration.
- Loads the model weights from the checkpoint file using `torch.load(..., weights_only=True)`.
- Applies the loaded state dictionary to the newly created model.

This method ensures that evaluation always uses the exact architecture and hyperparameters recorded 
during training, while loading only the model weights from the checkpoint.

#### `_construct_test_set(feature_dir, label_dir)`
- Resolves and sorts all feature and label chip paths.  
- Validates that both directories contain the same number of `.tif` files.
- Returns feature and label chip pairs

#### `_construct_dataloader(batch_size, num_workers)`
- Loads the feature and label chip pairs into a PyTorch DataLoader.

#### `run()`
- Builds a PyTorch Lightning Trainer using parameters from `TestConfig` (computing precision, device selection, logging).
- Loads the trained checkpoint and associated hyperparameters.
- Runs evaluation using `trainer.test()`.
- Logs all metrics to `lightning_logs/test_logs/` via `CSVLogger` and prints final evaluation results.

---

## Class: `Main`
Provides the command-line interface for running model evaluation.

### Arguments
**Test Arguments** (via `add_test_args` in `test_args.py`)

| Argument                | Type | Default     | Description                                            |
|-------------------------|------|-------------|--------------------------------------------------------|
| `model_checkpoint_path` | Path | –           | Path to the trained model checkpoint (`.ckpt`).        |
| `hparams_path`          | Path | –           | Path to the `hparams.yaml` file saved during training. |
| `feature_dir`           | Path | –           | Directory containing test feature chips.               |
| `label_dir`             | Path | –           | Directory containing test label chips.                 |
| `--batch_size`          | int  | `4`         | Batch size for evaluation.                             |
| `--num_workers`         | int  | `0`         | Number of CPU workers for data loading.                |
| `--compute_precision`   | str  | `"32-true"` | Computation precision (`16-mixed`, `32-true`, etc.).   |


---

## Output
Test produces:
- `metrics.csv` storing the evaluation metrics.
- `hparams.yaml` file storing the hyperparameters and configuration used during evaluation.
- Printed output summarizing final evaluation metrics such as loss, MCC, and other metrics.

Example directory structure:
```
script_root\
└── lightning_logs\
    └── test_logs\
        └── version_0\
            ├── metrics.csv
            └── hparams.yaml
```

---

## Example Usage
```bash
python test.py   ./models/model_01.ckpt   ./models/model_01.yaml   ./dataset_output/test_data/feature_chips   ./dataset_output/test_data/label_chips   --batch_size 4   --compute_precision 16-mixed
```

Both **relative** and **absolute** paths are supported for all inputs.

---

## Dependencies
- **Albumentations**: ensures consistent preprocessing during testing.  
- **model.py**: defines the `LightningDitchNet` architecture used for evaluation.  
- **preprocessing.py**: provides `DitchDataset`, which handles feature–label loading.  
- **PyTorch Lightning**: provides the structured evaluation workflow and logging.  
- **Segmentation Models PyTorch (smp)**: defines the model architecture and encoder.

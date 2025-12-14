# `train.py` — Training Pipeline for `LightningDitchNet`

## Overview
`train.py` handles the **training process** for the `LightningDitchNet` segmentation model. It loads the preprocessed 
feature and label chips produced by `preprocessing.py`, builds the dataset, defines augmentations, 
and trains the model using PyTorch Lightning.

---

## Class: `Train`
Encapsulates all components required to train `LightningDitchNet` end-to-end: 
model initialization, dataset preparation, augmentations, dataloaders, 
callbacks, and PyTorch Lightning trainer configuration.

### Initialization
The class is most commonly initialized through the CLI (`Main` class) or via the 
`DEM2Ditch` application, and its underlying structure is defined using the 
`TrainConfig` (containing a nested `ModelConfig`) from `config.py` as follows:

```python
Train(config: TrainConfig)
```

All configuration attributes are documented in the `Main` section, 
where each parameter is explained in a table.

### Main Responsibilities
- Initializes the `LightningDitchNet` model using hyperparameters defined in `ModelConfig`.
- Loads feature and label chips using `DitchDataset` from `preprocessing.py`.
- Splits data into training and validation sets with reproducible seeding.
- Defines augmentation and preprocessing transforms with Albumentations.
- Constructs PyTorch DataLoaders for both splits.
- Sets up checkpointing and optional early stopping callbacks.
- Builds and runs a PyTorch Lightning Trainer instance for full training execution.

---

### Methods

#### `_construct_train_val_sets(feature_dir, label_dir)`
- Resolves and sorts all feature and label chip paths.  
- Validates that both directories contain the same number of `.tif` files.  
- Splits data into 80% training and 20% validation sets using a fixed random seed (`random_state=14`) for reproducibility.

#### `_construct_transforms()`
- Defines two Albumentations transformation pipelines:
  - **Training transform:** includes random flips, rotations, and transpositions to increase dataset diversity.  
  - **Validation transform:** applies only conversion to tensor for consistent evaluation.

#### `_construct_dataloaders(batch_size, num_workers)`
- Wraps both datasets (`training` and `validation`) into `DataLoader` objects.  
- Enables shuffling for training data, memory pinning, and parallel workers for faster loading.

#### `_set_callbacks()`
Configures the training callbacks defined by TrainConfig:

- **ModelCheckpoint**
  - Saves the top-k performing checkpoints based on the monitored metric.

- **EarlyStopping** (optional)
  - Stops training if the monitored metric does not improve for the configured patience value.

#### `_is_weights_only_checkpoint(ckpt_path)`
Detects whether a checkpoint contains only model weights.
If so, training resumes in **fine-tuning mode**, reconstructing the model and loading weights manually.

#### `run()`
Executes the full training loop:
- Builds a Lightning `Trainer` using parameters from `TrainConfig` (epochs, computing precision, device selection, etc.).
- Starts training either from scratch or by resuming from a provided checkpoint.
- Logs all metrics and hyperparameters using a `CSVLogger` under `lightning_logs/train_logs/`.

<div style="border: 1.5px solid #d3d3d3; border-radius: 6px; padding: 10px;">

⚠️ **Using Pretrained Weights and Checkpoints** ⚠️

When a checkpoint is provided through `ckpt_path`, the training behavior depends on 
whether the checkpoint contains only model weights or a full training state.
This is detected automatically via `_is_weights_only_checkpoint()`.

---

**Weights-only checkpoint**

A checkpoint is considered weights-only when it includes:
- the model’s `state_dict` (model weights)

and does not include:
- optimizer state
- scheduler state
- epoch/step counters
- any other training metadata

If a weights-only checkpoint is used:
- A fresh `LightningDitchNet` model is initialized.
- Only the network weights are loaded.
- Training proceeds in **fine-tuning mode**, meaning the optimizer and learning rate scheduler 
are reinitialized and training restarts from epoch 0.

This approach is ideal when:
  - using pretrained weights,
  - transferring weights into a new training configuration,
  - continuing training when only weights were saved.

---

**Full checkpoint**

A full checkpoint contains:
- model weights
- optimizer state
- learning rate scheduler state
- training progress (epoch, step)
- all trainer metadata

If a full checkpoint is used:
- The entire training state is restored exactly as it was.
- The Trainer resumes from the recorded epoch and step.
- Optimizer, scheduler, and all internal states continue from their previous values.

This mode is intended for:
- seamlessly resuming long training runs,
- recovering after interruptions.

</div>

---

## Class: `Main`
Provides the command-line interface for running training, including all core training parameters 
as well as the model and scheduler configuration options.

### Arguments
**Training Arguments** (via `add_training_args` in `training_args.py`)

| Argument                | Type  | Default      | Description                                                            |
|-------------------------|-------|--------------|------------------------------------------------------------------------|
| `feature_dir`           | Path  | –            | Directory containing input feature chips.                              |
| `label_dir`             | Path  | –            | Directory containing label (mask) chips.                               |
| `max_epochs`            | int   | –            | Maximum number of training epochs.                                     |
| `--ckpt_path`           | Path  | `None`       | Optional checkpoint file to resume or fine-tune from.                  |
| `--val_size`            | float | `0.2`        | Fraction of samples used for validation.                               |
| `--batch_size`          | int   | `4`          | Batch size for training.                                               |
| `--num_workers`         | int   | `0`          | Number of parallel CPU workers for data loading.                       |
| `--compute_precision`   | str   | `"32-true"`  | Training precision (`16-mixed`, `bf16-true`, `32-true`, etc.).         |
| `--full_checkpoint`     | flag  | `False`      | If set, saves full training state instead of weights-only checkpoints. |
| `--save_top_k`          | int   | `10`         | Number of top checkpoints to keep based on monitored metric.           |
| `--checkpoint_monitor`  | str   | `"val_mcc"`  | Metric used to determine best checkpoints.                             |
| `--checkpoint_mode`     | str   | `"max"`      | Optimization direction for checkpoint metric (`min` or `max`).         |
| `--no_early_stop`       | flag  | `False`      | Disable early stopping.                                                |
| `--early_stop_patience` | int   | `50`         | Epochs with no improvement before early stopping triggers.             |
| `--early_stop_monitor`  | str   | `"val_loss"` | Metric monitored for early stopping.                                   |
| `--early_stop_mode`     | str   | `"min"`      | Optimization direction for early stop metric.                          |

**Model Arguments** (via `add_model_args` in `model_args.py`)

| Argument          | Type  | Default             | Description                                          |
|-------------------|-------|---------------------|------------------------------------------------------|
| `--encoder_name`  | str   | `"efficientnet-b4"` | Encoder backbone from `segmentation_models_pytorch`. |
| `--pos_weight`    | float | `3`                 | Weight for the positive ditch class in BCE loss.     |
| `--learning_rate` | float | `1e-4`              | Learning rate for the optimizer.                     |
| `--in_channels`   | int   | `2`                 | Number of input channels to the model.               |
| `--weight_decay`  | float | `1e-4`              | Weight decay (L2 regularization) for AdamW.          |

**Scheduler Arguments** (via `add_scheduler_args` in `model_args.py`)

| Argument                     | Type  | Default      | Description                                                      |
|------------------------------|-------|--------------|------------------------------------------------------------------|
| `--no_scheduler`             | flag  | `False`      | Disable learning rate scheduler.                                 |
| `--scheduler_monitor`        | str   | `"val_loss"` | Metric name monitored by the scheduler.                          |
| `--scheduler_mode`           | str   | `"min"`      | Improve-direction (`min` or `max`).                              |
| `--scheduler_factor`         | float | `0.5`        | Learning rate reduction factor.                                  |
| `--scheduler_patience`       | int   | `5`          | Epochs without improvement before learning rate reduction.       |
| `--scheduler_cooldown`       | int   | `5`          | Cooldown period after learning rate reduction.                   |
| `--scheduler_min_lr`         | float | `1e-7`       | Minimum possible learning rate.                                  |
| `--scheduler_threshold`      | float | `1e-3`       | Minimum improvement required to trigger learning rate reduction. |
| `--scheduler_threshold_mode` | str   | `"rel"`      | Threshold interpretation (`rel` or `abs`).                       |

---

## Output
Training produces:
- Training and validation metrics for each epoch.
- `hparams.yaml` file storing the hyperparameters and configuration used during training.
- Model checkpoints (`.ckpt`) saved according to the checkpointing strategy.

Example directory structure:
```
script_root\
└── lightning_logs\
    └── train_logs\
        └── version_0\
            ├── metrics.csv
            ├── hparams.yaml
            └── checkpoints\
                ├── epoch=0-step=10.ckpt
                ├── epoch=1-step=20.ckpt
                └── ...
```

---

## Example Usage
```bash
python train.py   ./dataset_output/training_data/feature_chips   ./dataset_output/training_data/label_chips   50   --encoder_name efficientnet-b4   --pos_weight 3   --batch_size 8   --compute_precision 16-mixed
```

Both **relative** and **absolute** paths are supported for all inputs.  

---

## Dependencies
- **Albumentations**: for data augmentation and preprocessing.
- **model.py**: defines the `LightningDitchNet` architecture used for training.
- **preprocessing.py**: provides `DitchDataset`, which handles feature–label loading.
- **PyTorch Lightning**: for structured training, validation, logging, and checkpointing.
- **scikit-learn**: used for dataset splitting (`train_test_split`).
- **Segmentation Models PyTorch (smp)**: defines the model architecture and encoder.

The trained model checkpoints are later used by **`test.py`** and **`inference.py`** for evaluation and prediction.

# `model.py` — LightningDitchNet Segmentation Model

`model.py` defines `LightningDitchNet`, a PyTorch Lightning–based U-Net segmentation model used to predict ditch probability maps from 2-channel input features (HPMF + ISI).  

The module includes the model architecture, loss function, evaluation metrics, and optimizer/scheduler configuration.

---

## Class: `LightningDitchNet`

### Overview  
A LightningModule wrapping a **U-Net** architecture from `segmentation_models_pytorch`.  
Designed for binary semantic segmentation with strong class imbalance.

### Initialization  
The class is most commonly initialized through the training pipeline (`train.py`), 
and its structure is defined using the `ModelConfig` from `config.py` as follows:”

```python
LightningDitchNet(config: ModelConfig)
```

All configuration attributes are documented in the `Main` section of the `train_documentation.md`, 
where each parameter is explained in the **"Model Arguments"** and **"Scheduler Arguments"** tables.

During initialization, the class:

- Builds a U-Net with the specified encoder. 
- Sets the number of input channels.
- Configures **weighted BCEWithLogitsLoss** to handle the highly imbalanced ditch vs. background classes.  
- Registers common binary classification metrics:  
  - Accuracy 
  - Recall 
  - Precision
  - Matthews Correlation Coefficient (MCC) 
  - Confusion-matrix stats  
- Stores hyperparameters for checkpointing and reproducibility.

---

### Methods

### `forward(x)`
- Runs a forward pass through the U-Net model and returns raw logits.  
- Used during training, validation, testing, and inference.

---

### `_shared_step(batch, stage)`
Shared computation used by all training phases. The method:

- Receives a batch `(features, labels)`.  
- Computes logits and sigmoid probabilities.
- Calculates weighted BCE loss.
- Computes accuracy, recall, precision, and MCC.
- Logs loss and metrics under phase-prefixed names (`train_*`, `val_*`, `test_*`).
- Logs confusion-matrix components (TP, FP, TN, FN) for validation and test.
- Returns the loss value.

---

### `training_step(batch, batch_idx)`
Runs the shared step in **training** mode.

### `validation_step(batch, batch_idx)`
Runs the shared step in **validation** mode and logs validation metrics.

### `test_step(batch, batch_idx)`
Runs the shared step in **test** mode and logs test metrics.

---

### `configure_optimizers()`
Defines the training optimization strategy.
- Uses AdamW with weight decay for stability.
- If `use_scheduler=False` (from `ModelConfig`), returns only the optimizer.
- If the scheduler is enabled:
  - Configures a **ReduceLROnPlateau** scheduler.
  - Monitors the metric defined in `scheduler_monitor`.
  - Allows the learning rate to decrease automatically when progress plateaus.
  - Supports customizable:
    - mode (`min` or `max`)
    - factor
    - patience
    - cooldown
    - minimum learning rate
    - threshold and threshold mode
- Returns both the optimizer and scheduler in the Lightning-compatible format.

---

## Dependencies
- **PyTorch** – model architecture, autograd, optimization  
- **PyTorch Lightning** – handles the training loop, checkpointing, and logging
- **segmentation_models_pytorch** – U-Net implementation with configurable encoders  
- **torchmetrics** – binary evaluation metrics  
- **PyTorch scheduler** – ReduceLROnPlateau for LR control

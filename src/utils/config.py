from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """
    Configuration for DEM preprocessing and chip generation.

    This configuration defines how raw DEM data is converted into 512×512
    training or testing chips. The process includes generating HPMF and ISI
    feature layers, rasterizing ditch vectors
    and saving the resulting feature and label tiles.

    Args:
        input_dem_dir: Directory containing raw DEM tiles (.tif).
        label_vector_data: Vector dataset (e.g., Shapefile/GPKG) containing ditch geometries.
        output_dir: Directory for saving generated chips and temporary rasters.
        mode: Whether preprocessing is generating chips for training or testing.
        ditch_width: Determines how wide the ditch features appear in the generated label mask.
        label_hpmf_threshold: Threshold for masking ditch areas based on HPMF depth.
    """

    input_dem_dir: Path
    label_vector_data: Path
    output_dir: Path
    mode: Literal["train", "test"] = "train"
    ditch_width: float = 1.5
    label_hpmf_threshold: float = -0.075


@dataclass
class ModelConfig:
    """
    Hyperparameter configuration for the LightningDitchNet model.

    This configuration contains all model-level settings including the encoder
    backbone, loss weighting, optimization parameters, and learning rate scheduler
    behaviour. These values primarily affect the model architecture and the
    training dynamics governed by the optimizer.

    Args:
        encoder_name: Encoder backbone used in the U-Net model.
        pos_weight: Weight applied to positive pixels in BCEWithLogitsLoss.
        lr: Learning rate used by the optimizer.
        in_channels: Number of input channels (commonly 2: HPMF + ISI).
        weight_decay: Weight decay value for AdamW optimizer.

        use_scheduler: Enable ReduceLROnPlateau learning rate scheduling.
        scheduler_monitor: Metric name to monitor (e.g., "val_loss", "val_mcc").
        scheduler_mode: Direction of improvement ("min" or "max").
        scheduler_factor: Factor by which the learning rate is reduced.
        scheduler_patience: Epochs without improvement before reducing LR.
        scheduler_cooldown: Cooldown period after LR is reduced.
        scheduler_min_lr: Lower bound of the learning rate.
        scheduler_threshold: Minimum metric improvement required to trigger LR reduction.
        scheduler_threshold_mode: Interpretation of threshold ("rel" or "abs").
    """

    encoder_name: str = "efficientnet-b4"
    pos_weight: float = 3.0
    lr: float = 1e-4
    in_channels: int = 2
    weight_decay: float = 1e-4

    use_scheduler: bool = True
    scheduler_monitor: Literal["train_loss", "train_acc", "train_recall", "train_prec",
                               "train_mcc", "val_loss", "val_acc", "val_recall",
                               "val_prec", "val_mcc"] = "val_loss"

    scheduler_mode: Literal["min", "max"] = "min"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_cooldown: int = 5
    scheduler_min_lr: float = 1e-7
    scheduler_threshold: float = 1e-3
    scheduler_threshold_mode: Literal["rel", "abs"] = "rel"


@dataclass
class TrainConfig:
    """
    Configuration for training the LightningDitchNet model.

    This configuration controls dataset loading, batching, validation split,
    model reproducibility, and precision settings. It also contains a nested
    ModelConfig instance that defines all hyperparameters used by the model
    and optimizer.

    Args:
        feature_dir: Directory containing feature chips used for training.
        label_dir: Directory containing corresponding label chips.
        max_epochs: Maximum number of training epochs.
        val_size: Fraction of samples used for validation.
        batch_size: Training batch size.
        num_workers: Number of CPU worker processes for DataLoader.
        compute_precision: Mixed/full precision mode used by Lightning.

        save_weights_only: Whether to store only model weights in checkpoint files.
        save_top_k: Number of the best checkpoints to keep (based on monitored metric).
        checkpoint_monitor: Metric name used for selecting the best checkpoints.
        checkpoint_mode: Optimization direction for the monitored metric ("min" or "max").

        early_stop_patience: Number of epochs with no improvement before stopping training.
        early_stop_monitor: Metric name that controls early stopping.
        early_stop_mode: Optimization direction for the monitored metric ("min" or "max").


        model_config: Nested ModelConfig with network hyperparameters.
    """

    feature_dir: Path
    label_dir: Path
    max_epochs: int
    ckpt_path: Path = None
    val_size: float = 0.2
    batch_size: int = 4
    num_workers: int = 0
    compute_precision: str = "32-true"

    save_weights_only: bool = True
    save_top_k: int = 10
    checkpoint_monitor: Literal["train_loss", "train_acc", "train_recall", "train_prec",
                                "train_mcc", "val_loss", "val_acc", "val_recall",
                                "val_prec", "val_mcc"] = "val_mcc"

    checkpoint_mode: Literal["min", "max"] = "max"

    use_early_stop: bool = True
    early_stop_patience: int = 50
    early_stop_monitor: Literal["train_loss", "train_acc", "train_recall", "train_prec",
                                "train_mcc", "val_loss", "val_acc", "val_recall",
                                "val_prec", "val_mcc"] = "val_loss"

    early_stop_mode: Literal["min", "max"] = "min"

    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class TestConfig:
    """
    Configuration for testing/evaluating a trained LightningDitchNet model.

    Defines checkpoint loading, test dataset paths, and evaluation precision.

    Args:
        model_checkpoint_path: Path to the model checkpoint (.ckpt) to evaluate.
        hparams_path: Path to the hparams.yaml file (including hyperparameters) saved during training.
        feature_dir: Directory containing feature chips for evaluation.
        label_dir: Directory containing corresponding label chips.
        batch_size: Batch size for evaluation.
        num_workers: Number of CPU worker processes for DataLoader.
        compute_precision: Mixed/full precision mode used by Lightning.
    """

    model_checkpoint_path: Path
    hparams_path: Path
    feature_dir: Path
    label_dir: Path
    batch_size: int = 4
    num_workers: int = 0
    compute_precision: Literal["16-true", "16-mixed", "bf16-true", "bf16-mixed",
                               "32-true", "64-true", "64", "32", "16", "bf16"] = "32-true"


@dataclass
class InferenceConfig:
    """
    Configuration for running model inference on DEM tiles.

    Specifies all required paths and runtime options for producing ditch
    probability, binary, and/or depth maps from DEM data.

    Args:
        model_dir: Directory containing one or more trained model checkpoints.
        input_dem_dir: Directory containing raw DEM tiles to process.
        output_dir: Directory where all inference outputs are written.

        threshold: Probability threshold used to create binary maps.
        output_prob_map: Whether to save probability rasters.
        output_binary_map: Whether to save thresholded rasters.
        output_depth_map: Whether to save derived depth rasters.
        device: Computation device ("cpu", "cuda", or auto-detection).
    """

    model_dir: Path
    input_dem_dir: Path
    output_dir: Path
    threshold: float = 0.3
    output_prob_map: bool = True
    output_binary_map: bool = True
    output_depth_map: bool = True
    device: Literal["cpu", "cuda", "auto"] = "auto"

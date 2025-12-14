def add_training_args(parser):
    group = parser.add_argument_group("Training options")

    group.add_argument("feature_dir", help="Path to directory containing input feature images.")
    group.add_argument("label_dir", help="Path to directory containing label images.")
    group.add_argument("max_epochs", type=int, help="Maximum number of training epochs to run.")

    group.add_argument("--ckpt_path",
                       default=None,
                       help="Optional path to a checkpoint for resuming training or fine-tuning. "
                            "By default, training starts from scratch.")

    group.add_argument("--val_size",
                       type=float,
                       default=0.2,
                       help="Fraction of samples used for validation.")

    group.add_argument("--batch_size",
                       type=int,
                       default=4,
                       help="Batch size for training.")

    group.add_argument("--num_workers",
                       type=int,
                       default=0,
                       help="Number of parallel CPU workers used for loading batches from disk.")

    group.add_argument("--compute_precision",
                       choices=["16-true", "16-mixed",
                                "bf16-true", "bf16-mixed",
                                "32-true", "64-true",
                                "64", "32", "16", "bf16"],

                       default="32-true",
                       help="Computation precision for training. More info: "
                            "https://lightning.ai/docs/pytorch/stable/common/precision_basic.html")

    group.add_argument("--full_checkpoint",
                       dest="save_weights_only",
                       action="store_false",
                       help="If set, save the full training state instead of weights-only checkpoints.")

    group.add_argument("--save_top_k",
                       type=int,
                       default=10,
                       help="Number of top checkpoints to keep based on the monitored metric.")

    group.add_argument("--checkpoint_monitor",
                       type=str,
                       choices=["train_loss", "train_acc", "train_recall", "train_prec", "train_mcc",
                                "val_loss", "val_acc", "val_recall", "val_prec", "val_mcc"],

                       default="val_mcc",
                       help="Metric name used to determine which checkpoints are considered best.")

    group.add_argument("--checkpoint_mode",
                       choices=["min", "max"],
                       default="max",
                       help="Optimization direction for the monitored metric ('min' or 'max').")

    group.add_argument("--no_early_stop",
                       dest="use_early_stop",
                       action="store_false",
                       help="Disable early stopping during training.")

    group.add_argument("--early_stop_patience",
                       type=int,
                       default=50,
                       help="Number of epochs with no improvement before early stopping triggers.")

    group.add_argument("--early_stop_monitor",
                       type=str,
                       choices=["train_loss", "train_acc", "train_recall", "train_prec", "train_mcc",
                                "val_loss", "val_acc", "val_recall", "val_prec", "val_mcc"],

                       default="val_loss",
                       help="Metric to monitor for early stopping.")

    group.add_argument("--early_stop_mode",
                       type=str,
                       default="min",
                       choices=["min", "max"],
                       help="Direction in which the monitored metric is optimized.")


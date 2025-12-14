import segmentation_models_pytorch as smp


def add_model_args(parser):
    group = parser.add_argument_group("Model options")

    group.add_argument("--encoder_name",
                       default="efficientnet-b4",
                       choices=smp.encoders.get_encoder_names(),
                       help="Encoder backbone for DitchNet. "
                            "Choices: https://smp.readthedocs.io/en/latest/encoders.html")

    group.add_argument("--pos_weight",
                       type=float,
                       default=3,
                       help="Weighting factor for positive (ditch) class in the BCE loss to handle imbalance.")

    group.add_argument("--learning_rate",
                       type=float,
                       default=1e-4,
                       help="Learning rate for optimizer.")

    group.add_argument("--in_channels",
                       type=int,
                       default=2,
                       help="Number of input channels for the model.")

    group.add_argument("--weight_decay",
                       type=float,
                       default=1e-4,
                       help="Weight decay for the optimizer (L2 regularization).")


def add_scheduler_args(parser):
    group = parser.add_argument_group("Scheduler options")

    group.add_argument("--no_scheduler",
                       dest="use_scheduler",
                       action="store_false",
                       help="Disable learning-rate scheduler.")

    group.add_argument("--scheduler_monitor",
                       type=str,
                       default="val_loss",
                       choices=["train_loss", "train_acc", "train_recall", "train_prec",
                                "train_mcc", "val_loss", "val_acc", "val_recall",
                                "val_prec", "val_mcc"],

                       help="Metric name to monitor for learning rate scheduling.")

    group.add_argument("--scheduler_mode",
                       type=str,
                       default="min",
                       choices=["min", "max"],
                       help="ReduceLROnPlateau mode.")

    group.add_argument("--scheduler_factor",
                       type=float,
                       default=0.5,
                       help="Factor by which to reduce the learning rate.")

    group.add_argument("--scheduler_patience",
                       type=int,
                       default=5,
                       help="Epochs with no improvement before reducing learning rate.")

    group.add_argument("--scheduler_cooldown",
                       type=int,
                       default=5,
                       help="Cooldown epochs after learning-rate reduction.")

    group.add_argument("--scheduler_min_lr",
                       type=float,
                       default=1e-7,
                       help="Minimum learning rate allowed.")

    group.add_argument("--scheduler_threshold",
                       type=float,
                       default=1e-3,
                       help="Improvement threshold to trigger learning rate reduction.")

    group.add_argument("--scheduler_threshold_mode",
                       type=str,
                       default="rel",
                       choices=["rel", "abs"],
                       help="Threshold mode for learning rate scheduler.")

def add_test_args(parser):
    parser.add_argument("model_checkpoint_path",
                        help="Path to the trained LightningDitchNet model checkpoint (.ckpt file).")

    parser.add_argument("hparams_path",
                        help="Path to the hparams.yaml file saved during training.")

    parser.add_argument("feature_dir", help="Path to directory containing input feature images.")
    parser.add_argument("label_dir", help="Path to directory containing label (mask) images.")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing.")

    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of parallel CPU workers used for loading batches from disk.")

    parser.add_argument("--compute_precision",
                        choices=["16-true", "16-mixed",
                                 "bf16-true", "bf16-mixed",
                                 "32-true", "64-true",
                                 "64", "32", "16", "bf16"],

                        default="32-true",
                        help="Computation precision for testing. More information: "
                             "https://lightning.ai/docs/pytorch/stable/common/precision_basic.html")

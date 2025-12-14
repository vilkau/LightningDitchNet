def add_inference_args(parser):
    parser.add_argument("model_dir",
                        help="Directory containing one or more LightningDitchNet model checkpoints (*.ckpt).")

    parser.add_argument("input_dem_dir", help="Directory containing DEM files (.tif) to process.")
    parser.add_argument("output_dir", help="Directory where output maps will be saved.")

    parser.add_argument("--threshold",
                        type=float,
                        default=0.3,
                        help="Binarization threshold for the output map.")

    parser.add_argument("--no_prob_map",
                        dest="output_prob_map",
                        action="store_false",
                        help="Disable saving of the probability map output (enabled by default).")

    parser.add_argument("--no_binary_map",
                        dest="output_binary_map",
                        action="store_false",
                        help="Disable saving of the binary map output (enabled by default).")

    parser.add_argument("--no_depth_map",
                        dest="output_depth_map",
                        action="store_false",
                        help="Disable saving of the depth map output (enabled by default).")

    parser.add_argument("--device",
                        choices=["cpu", "cuda", "auto"],
                        default="auto",
                        help='Computation device: "cpu", "cuda", or "auto" (automatically detect GPU if available).')

import argparse

import torch

from pathlib import Path
import shutil
import numpy as np

from skimage.transform import resize
import rasterio

from utils.tools import (minmax_normalized_image,
                         create_hpmf_layer,
                         create_isi_layer,
                         create_feature_layer,
                         fetch_hparams_from_yaml)

from model import LightningDitchNet

from utils.config import InferenceConfig, ModelConfig
from utils.cli_args.inference_args import add_inference_args

from osgeo import gdal
gdal.UseExceptions()


class Inference:
    def __init__(self, config: InferenceConfig):
        self.config = config

        if not config.output_prob_map and not config.output_binary_map and not config.output_depth_map:
            raise ValueError('At least one of "output_prob_map", "output_binary_map" or '
                             '"output_depth_map" must be True.')

        # Define directories
        self.input_dem_dir = Path(config.input_dem_dir).resolve()
        self.output_dir = Path(config.output_dir).resolve()

        self.device = torch.device(self._init_device())
        print(f"\nUsing device: {self.device}\n")

        # Initialize selected model(s)
        self.models = []
        self._init_models(config.model_dir)

        # Initialize output and temp directories
        self.output_probability_dir, self.output_binary_dir, self.output_depth_dir = None, None, None
        self._set_output_directories()

        self.temp_dir, self.hpmf_temp, self.isi_temp = None, None, None
        self._set_temporary_directories()

        self.invalid_inputs = []

    def _init_device(self):
        # Auto-select computation device
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        return self.config.device

    def _init_models(self, model_dir):
        # Resolve the path and ensure at least one .ckpt file exists
        model_dir = Path(model_dir).resolve()
        if not any(model_dir.glob("*.ckpt")):
            raise ValueError(f"No model checkpoints (*.ckpt) found in directory: {model_dir}")

        print("The following models will be used and their predictions will be averaged:")

        # Load each checkpoint, move model to the selected device, store it
        for ckpt_path in model_dir.glob("*.ckpt"):
            encoder_name, in_channels = fetch_hparams_from_yaml(mode="inference",
                                                                yaml_path=model_dir / f"{ckpt_path.stem}.yaml")

            model_config = ModelConfig(encoder_name=encoder_name, in_channels=in_channels)

            model = LightningDitchNet(model_config)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])

            self.models.append(model.to(self.device))

            print(ckpt_path.name)

        print("")

        # Set all models to evaluation mode (disables training-specific layers and gradients)
        for model in self.models:
            model.eval()

    def _set_output_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories only for the enabled output types
        if self.config.output_prob_map:
            print("Probability map output: enabled")
            self.output_probability_dir = self.output_dir / "probability_maps"
            self.output_probability_dir.mkdir(parents=True, exist_ok=True)

        if self.config.output_binary_map:
            print(f"Binary map output: enabled (threshold: {self.config.threshold})")
            self.output_binary_dir = self.output_dir / "binary_maps"
            self.output_binary_dir.mkdir(parents=True, exist_ok=True)

        if self.config.output_depth_map:
            print("Depth map output: enabled")
            self.output_depth_dir = self.output_dir / "depth_maps"
            self.output_depth_dir.mkdir(parents=True, exist_ok=True)

    def _set_temporary_directories(self):
        # Create temporary working directory for intermediate raster products
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.hpmf_temp = self.temp_dir / "hpmf_temp.tif"
        self.isi_temp = self.temp_dir / "isi_temp.tif"

    def _create_output_layer(self, feature_array, orig_height, orig_width, resampled_height, resampled_width):
        chip_size = 512

        output_array = np.empty((resampled_height, resampled_width), dtype=np.float32)

        # Iterate over the raster in 512×512 tiles, last tiles may overlap to fully cover the image
        for i in range(0, resampled_height, chip_size):
            start_i = min(i, resampled_height - chip_size)
            for j in range(0, resampled_width, chip_size):
                start_j = min(j, resampled_width - chip_size)

                # Extract 2-channel feature tile (HPMF + ISI)
                feature_chip = feature_array[:, start_i:start_i + chip_size, start_j:start_j + chip_size]

                # Add batch dimension and send to device
                feature_chip = feature_chip[np.newaxis, :, :]
                feature_tensor = torch.from_numpy(feature_chip).float().to(self.device)

                predictions = []

                # Run inference for all models with gradients disabled (faster, lower memory use)
                with torch.no_grad():
                    for model in self.models:
                        # Forward pass through the current model
                        predicted = model(feature_tensor)

                        # Convert logits to probabilities and move result to CPU as NumPy array
                        predicted = torch.sigmoid(predicted).squeeze().cpu().numpy()

                        # Store each model's prediction for later averaging
                        predictions.append(predicted)

                # Average predictions from all models to produce the ensemble output
                merged_prediction = np.mean(predictions, axis=0)

                # Place merged prediction back into the output mosaic
                output_array[start_i:start_i + chip_size, start_j:start_j + chip_size] = merged_prediction

        # Resample back to the original DEM resolution
        output_array = resize(output_array, (orig_height, orig_width), order=1, preserve_range=True, anti_aliasing=False)

        return output_array

    def _output_probability_map(self, input_path, profile, output_array):
        # Save continuous probability map
        profile.update(dtype=rasterio.float32, count=1, nodata=None)
        output_file = self.output_probability_dir / f"{input_path.stem}_ditch_probability.tif"
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(output_array, 1)

    def _output_binary_map(self, input_path, profile, output_array):
        # Convert probabilities to binary values using threshold
        filtered_output_array = (output_array >= self.config.threshold).astype(np.uint8)

        # Save binary map
        profile.update(dtype=rasterio.uint8, count=1, nodata=None)
        output_file = self.output_binary_dir / f"{input_path.stem}_ditch_binary.tif"
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(filtered_output_array, 1)

    def _output_depth_map(self, input_path, output_array):
        with rasterio.open(self.hpmf_temp) as src:
            hpmf_ref = src.read(1)
            profile = src.profile

        # Use HPMF depth where ditch probability > 0.1, replacing no-data with 0
        hpmf_ref[hpmf_ref == -9999] = 0
        output_array = np.where(output_array > 0.1, hpmf_ref, 0)

        # Force all positive elevations to 0 to keep only negative HPMF values (depressions)
        output_array[output_array > 0] = 0

        # Save HPMF depth map
        output_file = self.output_depth_dir / f"{input_path.stem}_ditch_depth.tif"
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(output_array, 1)

    def _process_single_dem(self, dem_path):
        with rasterio.open(dem_path) as src:
            orig_height = src.height
            orig_width = src.width
            profile = src.profile

        if orig_height < 500 or orig_width < 500:
            print(f"Input image {dem_path.name} is too small — minimum size is 500x500 pixels.")
            self.invalid_inputs.append(dem_path.name)
            return

        # Generate feature layers (HPMF and ISI) and normalize them
        hpmf_array = minmax_normalized_image(create_hpmf_layer(dem_path, self.hpmf_temp), constant_fill_value=1)
        isi_array = minmax_normalized_image(create_isi_layer(dem_path, self.isi_temp), constant_fill_value=0)

        # Slight upscaling to align with expected model resolution
        resampled_height = int(orig_height + (orig_height * 0.024))
        resampled_width = int(orig_width + (orig_width * 0.024))

        # Stack normalized features and perform model inference
        feature_array = create_feature_layer(hpmf_array, isi_array, resampled_height, resampled_width)
        output_array = self._create_output_layer(feature_array, orig_height, orig_width,
                                                 resampled_height, resampled_width)

        # Write selected outputs
        if self.config.output_prob_map:
            self._output_probability_map(dem_path, profile, output_array)

        if self.config.output_binary_map:
            self._output_binary_map(dem_path, profile, output_array)

        if self.config.output_depth_map:
            self._output_depth_map(dem_path, output_array)

        # Remove temporary files
        for temp in (self.hpmf_temp, self.isi_temp):
            if temp.exists():
                temp.unlink()

    @staticmethod
    def _output_virtual_raster(output_dir, output_name):
        raster_files = [raster_name for raster_name in output_dir.glob("*.tif")]

        vrt_path = output_dir / output_name
        gdal.BuildVRT(vrt_path, raster_files)

    def _create_virtual_rasters(self):
        # Build VRT mosaics from generated output rasters
        if self.config.output_prob_map:
            self._output_virtual_raster(self.output_probability_dir, "ditch_probability_map.vrt")

        if self.config.output_binary_map:
            self._output_virtual_raster(self.output_binary_dir, "ditch_binary_map.vrt")

        if self.config.output_depth_map:
            self._output_virtual_raster(self.output_depth_dir, "ditch_depth_map.vrt")

    def predict(self):
        print(f"\nRunning LightningDitchNet inference on DEM files in: {self.input_dem_dir}\n")

        dem_files = list(self.input_dem_dir.glob("*.tif"))
        if not dem_files:
            print("No DEM (.tif) files found — nothing to process.")
            if self.config.output_prob_map:
                shutil.rmtree(self.output_probability_dir)

            if self.config.output_binary_map:
                shutil.rmtree(self.output_binary_dir)

            if self.config.output_depth_map:
                shutil.rmtree(self.output_depth_dir)

            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

            return

        # Process all input DEMs
        for dem_path in dem_files:
            print(f"Processing: {dem_path.name}")
            self._process_single_dem(dem_path)

        self._create_virtual_rasters()

        # Remove the temporary working directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        print(f"\nAll predictions completed.")

        # Report skipped files
        if self.invalid_inputs:
            print(f"\nThe following input files failed to process:")
            for dem_name in self.invalid_inputs:
                print(dem_name)


class Main:
    def __init__(self):
        args = self._parse_arguments()
        inference_config = InferenceConfig(model_dir=args.model_dir,
                                           input_dem_dir=args.input_dem_dir,
                                           output_dir=args.output_dir,
                                           threshold=args.threshold,
                                           output_prob_map=args.output_prob_map,
                                           output_binary_map=args.output_binary_map,
                                           output_depth_map=args.output_depth_map,
                                           device=args.device)

        self.predictor = Inference(inference_config)
        self.run()

    @staticmethod
    def _parse_arguments():
        parser = argparse.ArgumentParser(description="Generate ditch probability, binary and depth maps "
                                                     "from DEM data using a trained LightningDitchNet models.")
        add_inference_args(parser)

        return parser.parse_args()

    def run(self):
        self.predictor.predict()


if __name__ == "__main__":
    Main()

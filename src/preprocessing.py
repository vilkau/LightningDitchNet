import argparse

from torch.utils.data import Dataset

from pathlib import Path
import numpy as np
import shutil

import geopandas as gpd
from shapely.geometry import box

import tifffile as tiff
import rasterio
from rasterio import features
from skimage.transform import resize

from utils.tools import (minmax_normalized_image,
                         create_hpmf_layer,
                         create_isi_layer,
                         create_feature_layer)

from utils.config import PreprocessingConfig
from utils.cli_args.preprocessing_args import add_preprocessing_args

from whitebox.whitebox_tools import WhiteboxTools
wbt = WhiteboxTools()
wbt.verbose = False


class DitchDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y

        # Albumentations transformation pipeline
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Read the TIFF file and convert to float32 for model input
        feature = tiff.imread(self.X[idx]).astype(np.float32)

        # Move channel dimension from [C, H, W] → [H, W, C] for Albumentations
        feature = np.moveaxis(feature, 0, -1)

        # Load binary label TIFF
        label = tiff.imread(self.y[idx]).astype(np.uint8)

        # Apply identical transformations to both feature and label
        augmented = self.transform(image=feature, label=label)
        feature = augmented["image"]
        label = augmented["label"]

        # Convert mask to binary (0 or 1), cast to float32, and add channel dimension
        # Resulting shape: [1, H, W]
        label = (label > 0).float().unsqueeze(0)

        return feature, label


class ChipGenerator:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

        # Resolve input/output paths
        self.input_dem_dir = Path(config.input_dem_dir).resolve()
        self.label_vector_data = config.label_vector_data
        self.output_dir = Path(config.output_dir).resolve()

        # Define and prepare directories
        self.data_dir, self.temp_dir = None, None
        self.hpmf_temp, self.isi_temp, self.label_temp = None, None, None
        self.feature_chip_dir, self.label_chip_dir = None, None
        self._set_directories()

    def _set_directories(self):
        # Select output folder based on mode
        if self.config.mode == "train":
            self.data_dir = self.output_dir / "training_data"
        elif self.config.mode == "test":
            self.data_dir = self.output_dir / "test_data"

        # Create standard folder structure for generated chips
        self.feature_chip_dir = self.data_dir / "feature_chips"
        self.label_chip_dir = self.data_dir / "label_chips"
        self.temp_dir = self.data_dir / "temp"

        for directory in [self.data_dir, self.temp_dir, self.feature_chip_dir, self.label_chip_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Temporary intermediate rasters
        self.hpmf_temp = self.temp_dir / "hpmf_temp.tif"
        self.isi_temp = self.temp_dir / "isi_temp.tif"
        self.label_temp = self.temp_dir / "label_temp.tif"

        self.invalid_inputs = []

    def _create_label_layer(self, dem_path, hpmf_array, resampled_height, resampled_width):
        with rasterio.open(dem_path) as src:
            dem_bounds = src.bounds
            dem_shape = src.shape
            transform = src.transform

        # Load ditch vector dataset (e.g., shapefile or GeoPackage) containing line geometries
        label_vector_gdf = gpd.read_file(self.label_vector_data)

        # Create a polygon covering the DEM tile extent
        dem_geom = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
        dem_gdf = gpd.GeoDataFrame(geometry=[dem_geom], crs=label_vector_gdf.crs)

        # Clip vector data (ditches) to DEM tile extent
        clipped_label_vector_gdf = gpd.clip(gdf=label_vector_gdf, mask=dem_gdf)

        # Buffer vector geometries with given width
        buffered_label_geom = clipped_label_vector_gdf.buffer(distance=self.config.ditch_width)

        # Rasterize buffered geometries onto DEM tile grid
        label_array = features.rasterize(shapes=[(geom, 1) for geom in buffered_label_geom.geometry],
                                         out_shape=dem_shape,  # Match output size to DEM raster
                                         transform=transform,  # Align to same grid/coordinates as DEM
                                         fill=0,               # Background pixels get value 0
                                         dtype=np.uint8,       # Use 8-bit integer values
                                         all_touched=True)     # Mark all pixels touched by geometry, not just centers

        # Keep only pixels inside the buffer where HPMF <= threshold
        label_array = np.where((label_array == 1) & (hpmf_array <= self.config.label_hpmf_threshold), 1, 0)

        # Save the binary label raster (0 = background, 1 = ditch)
        tiff.imwrite(self.label_temp, label_array.astype(np.uint8))

        # Apply majority filter to clean noise and smooth labels
        wbt.majority_filter(i=self.label_temp, output=self.label_temp, filterx=3, filtery=3)

        # Resample label raster to align with model resolution by
        # using nearest-neighbor interpolation (order=0) to preserve class values
        label_array = tiff.imread(self.label_temp)
        label_array = resize(label_array, (resampled_height, resampled_width),
                             order=0, preserve_range=True, anti_aliasing=False)

        return label_array

    def _generate_single_chip_pair(self, feature_array, label_array,
                                   start_i, start_j, chip_size,
                                   chip_idx, min_ditch_pixel_percentage=0.1):

        # Extract a chip from the label array
        label_chip = label_array[start_i:start_i + chip_size, start_j:start_j + chip_size]

        # Skip chip if less than 0.1% of pixels are ditch (to avoid mostly empty samples)
        if np.mean(label_chip == 1) * 100 < min_ditch_pixel_percentage:
            return

        # Save the label chip as 8-bit binary TIFF image
        label_chip_file = self.label_chip_dir / f"{chip_idx}.tif"
        tiff.imwrite(label_chip_file, label_chip.astype(np.uint8))

        # Extract matching feature chip (2-channel stack)
        feature_chip = feature_array[:, start_i:start_i + chip_size, start_j:start_j + chip_size]

        # Save feature chip as float32 TIFF image
        feature_chip_file = self.feature_chip_dir / f"{chip_idx}.tif"
        tiff.imwrite(feature_chip_file, feature_chip.astype(np.float32))

    def generate_chips(self):
        print(f"\nRunning LightningDitchNet preprocessing on DEM files in: {self.input_dem_dir}\n")

        print(f"Mode: {self.config.mode}")
        print(f"Ditch label width: {self.config.ditch_width}")
        print(f"Label HPMF threshold: {self.config.label_hpmf_threshold}\n")

        dem_files = list(self.input_dem_dir.glob("*.tif"))

        if not dem_files:
            print("No DEM (.tif) files found — nothing to process.")
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            return

        chip_idx = 0

        for dem_path in dem_files:
            print(f"Processing: {dem_path.name}")
            with rasterio.open(dem_path) as src:
                orig_height = src.height
                orig_width = src.width

            if orig_height < 500 or orig_width < 500:
                print(f"Input image {dem_path.name} is too small — minimum size is 500x500 pixels.")
                self.invalid_inputs.append(dem_path.name)
                continue

            # Slight upscaling to align with expected model resolution
            resampled_height = int(orig_height + (orig_height * 0.024))
            resampled_width = int(orig_width + (orig_width * 0.024))

            # Create HPMF (High-Pass Median Filter) layer and corresponding ditch label raster.
            # Label generation must use the original HPMF values for thresholding,
            # so normalization is applied only afterward for model input preparation.
            hpmf_array = create_hpmf_layer(dem_path, self.hpmf_temp)
            label_array = self._create_label_layer(dem_path, hpmf_array, resampled_height, resampled_width)

            # Normalize feature layers after label creation to standardize model inputs
            hpmf_array = minmax_normalized_image(hpmf_array, constant_fill_value=1)
            isi_array = minmax_normalized_image(create_isi_layer(dem_path, self.isi_temp), constant_fill_value=0)

            # Combine normalized HPMF and ISI into a 2-channel feature array
            feature_array = create_feature_layer(hpmf_array, isi_array, resampled_height, resampled_width)

            chip_size = 512

            # Iterate over the raster in 512×512 tiles, last tiles may overlap to fully cover the image
            for i in range(0, resampled_height, chip_size):
                start_i = min(i, resampled_height - chip_size)
                for j in range(0, resampled_width, chip_size):
                    start_j = min(j, resampled_width - chip_size)

                    self._generate_single_chip_pair(feature_array, label_array, start_i, start_j, chip_size, chip_idx)
                    chip_idx += 1

            # Remove temporary files
            self.hpmf_temp.unlink()
            self.isi_temp.unlink()
            self.label_temp.unlink()

        # Remove the temporary working directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        print(f"\nPreprocessing completed.")

        # Report skipped files
        if self.invalid_inputs:
            print(f"\nThe following input files failed to process:")
            for dem_name in self.invalid_inputs:
                print(dem_name)


class Main:
    def __init__(self):
        args = self._parse_arguments()
        config = PreprocessingConfig(args.input_dem_dir,
                                     args.label_vector_data,
                                     args.output_dir,
                                     mode=args.mode,
                                     ditch_width=args.ditch_width,
                                     label_hpmf_threshold=args.label_hpmf_threshold)

        self.chip_generator = ChipGenerator(config)
        self.run()

    @staticmethod
    def _parse_arguments():
        parser = argparse.ArgumentParser(description="Preprocess DEM data into 512×512 "
                                                     "feature and label chips for LightningDitchNet.")
        add_preprocessing_args(parser)

        return parser.parse_args()

    def run(self):
        self.chip_generator.generate_chips()


if __name__ == "__main__":
    Main()

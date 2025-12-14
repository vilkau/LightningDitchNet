import yaml
from typing import Literal

import numpy as np

from sklearn.preprocessing import MinMaxScaler

import tifffile as tiff
from skimage.transform import resize

from whitebox.whitebox_tools import WhiteboxTools
wbt = WhiteboxTools()
wbt.verbose = False


def minmax_normalized_image(image, constant_fill_value):
    # Return constant array if image has no variation
    if np.max(image) == np.min(image):
        return np.full(image.shape, constant_fill_value, dtype=np.float32)

    # Mask NaN and common no-data placeholders
    mask = np.isnan(image) | (image == -9999) | (image == -32768)
    valid = image[~mask]

    # Return constant array if image contains only NaN-values
    if valid.size == 0:
        return np.full(image.shape, constant_fill_value, dtype=np.float32)

    scaler = MinMaxScaler()

    # Fit MinMaxScaler on valid pixels and insert scaled values back into their original positions
    scaled = np.zeros_like(image, dtype=np.float32)
    scaled[~mask] = scaler.fit_transform(valid.reshape(-1, 1)).flatten()

    # Fill masked pixels with mean of valid values to avoid gaps
    scaled[mask] = np.mean(scaled[~mask])

    return scaled


def create_hpmf_layer(dem_path, hpmf_path):
    wbt.high_pass_median_filter(i=dem_path, output=hpmf_path, filterx=11, filtery=11)
    hpmf_array = tiff.imread(hpmf_path)

    return hpmf_array


def create_isi_layer(dem_path, isi_path):
    wbt.impoundment_size_index(dem=dem_path, damlength=6, out_max=isi_path)
    isi_array = tiff.imread(isi_path)

    return isi_array


def create_feature_layer(hpmf_array, isi_array, resampled_height, resampled_width):
    # Stack HPMF and ISI layers as channels: [2, H, W]
    feature_array = np.stack((hpmf_array, isi_array), axis=0)

    # Resize to match target chip resolution (bilinear interpolation)
    feature_array = resize(feature_array, (2, resampled_height, resampled_width),
                           order=1, preserve_range=True, anti_aliasing=False)
    return feature_array


def fetch_hparams_from_yaml(mode: Literal["test", "inference"], yaml_path):
    with open(yaml_path) as file:
        hyperparameters = yaml.safe_load(file)

    encoder_name, in_channels, pos_weight = (hyperparameters["encoder_name"],
                                             hyperparameters["in_channels"],
                                             hyperparameters["pos_weight"])
    if mode == "inference":
        return encoder_name, in_channels

    return encoder_name, in_channels, pos_weight

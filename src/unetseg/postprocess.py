import logging
import math
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import rasterio
import rasterio.merge
import rasterio.windows
from rasterio.transform import Affine
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def get_bounds_from_image_files(image_files: List[str]) -> Tuple[float]:
    """Get bounds from all images, and transform to pixels based on the affine transform"""
    xs = []
    ys = []
    for img_path in tqdm(image_files):
        with rasterio.open(img_path) as src:
            left, bottom, right, top = src.bounds
        xs.extend([left, right])
        ys.extend([bottom, top])
    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)
    return (dst_w, dst_s, dst_e, dst_n)


def crop_image(img: np.ndarray, margin_ratio: float) -> np.ndarray:
    """Center crop an image, with a margin of ``margin_ratio``"""
    h, w = img.shape[0], img.shape[1]
    h_margin = math.floor(h * margin_ratio)
    w_margin = math.floor(w * margin_ratio)
    return img[h_margin:-h_margin, w_margin:-w_margin]


def merge(images_dir: str, output_path: str, crop_margin_ratio: float = 0.125):
    """Merge all images in ``images_dir`` into a single image"""
    image_paths = glob(os.path.join(images_dir, "*.tif"))
    if not image_paths:
        raise RuntimeError("images_dir does not contain any .tif file")

    # Get the profile and affine of some image as template for output image
    with rasterio.open(image_paths[0]) as src:
        profile = src.profile.copy()
        src_res = src.res
        src_count = src.count
        # nodataval = src.nodatavals[0]
        # dt = src.dtypes[0]
        # transform = src.transform

    # Get bounds from all images, and transform to (width, height) in pixels
    dst_w, dst_s, dst_e, dst_n = get_bounds_from_image_files(image_paths)

    _logger.debug("Output bounds: %r", (dst_w, dst_s, dst_e, dst_n))
    output_transform = Affine.translation(dst_w, dst_n)
    _logger.debug("Output transform, before scaling: %r", output_transform)

    output_transform *= Affine.scale(src_res[0], -src_res[1])
    _logger.debug("Output transform, after scaling: %r", output_transform)

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    output_width = int(math.ceil((dst_e - dst_w) / src_res[0]))
    output_height = int(math.ceil((dst_n - dst_s) / src_res[1]))

    # Adjust bounds to fit
    dst_e, dst_s = output_transform * (output_width, output_height)
    _logger.debug("Output width: %d, height: %d", output_width, output_height)
    _logger.debug("Adjusted bounds: %r", (dst_w, dst_s, dst_e, dst_n))

    # Set width and height, and other attributes
    profile.update(
        width=output_width, height=output_height, transform=output_transform, tiled=True
    )

    # Call rasterio.merge using windowed reading-writing
    # and a custom callable that center-crops image.
    with rasterio.open(output_path, "w", **profile) as dst:
        for image_path in image_paths:
            with rasterio.open(image_path) as src:
                img = np.dstack(src.read())
                b = src.bounds
                window = rasterio.windows.from_bounds(
                    b[0], b[1], b[2], b[3], output_transform
                )
                # window = window.round_shape()

            cropped_img = crop_image(img, margin_ratio=crop_margin_ratio)
            # cropped_img = img.copy()

            for b in range(src_count):
                dst.write(cropped_img[:, :, b], b + 1, window=window)


if __name__ == "__main__":
    import sys

    merge(sys.argv[1], sys.argv[2])

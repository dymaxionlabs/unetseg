from itertools import zip_longest

import cv2
import skimage.transform
import tensorflow as tf
from tensorflow.keras.models import Model


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_model(model_path: str) -> Model:
    """Load model from ``model_path``"""
    return tf.keras.models.load_model(model_path)


def resize(image, size):
    """Resize multiband image to an image of size (h, w)"""
    n_channels = image.shape[2]
    if n_channels >= 4:
        return skimage.transform.resize(
            image, size, mode="constant", preserve_range=True
        )
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

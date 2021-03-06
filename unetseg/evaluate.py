import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage.transform import resize

from unetseg.predict import PredictConfig
from unetseg.train import TrainConfig, build_data_generator


def plot_data_generator(
    num_samples: int = 3,
    fig_size=(20, 10),
    *,
    train_config: TrainConfig,
    img_ch: int = 3
):
    """
    Plots some samples from a data generator.

    Parameters
    ----------
    num_samples : int
        Number of samples to plot.
    fig_size : tuple
        Figure size.
    img_ch : int
        Number of channels.
    train_config : TrainConfig
        Training configuration object.

    """

    if train_config.n_channels < 4:
        img_ch = train_config.n_channels

    images_dir = os.path.join(train_config.images_path, "images")
    mask_dir = train_config.masks_path or os.path.join(
        train_config.images_path, "extent"
    )

    images = glob(os.path.join(images_dir, "*.tif"))

    data_generator = build_data_generator(
        images, config=train_config, mask_dir=mask_dir
    )

    def plot_samples(plt, generator, num):
        j = 0
        for image_batch, mask_batch in data_generator:
            for image, mask in zip(image_batch, mask_batch):
                _, ax = plt.subplots(
                    nrows=1, ncols=train_config.n_classes + 1, figsize=fig_size
                )

                if train_config.n_channels < 4:
                    image = image[:, :, 0].astype(np.uint8)
                else:
                    image = image[:, :, :img_ch].astype(np.uint8)

                ax[0].imshow(image)
                for i in range(train_config.n_classes):
                    ax[i + 1].imshow(mask[:, :, i])
                j += 1
                if j >= num:
                    return

    plot_samples(plt, data_generator, num_samples)
    plt.show()


def plot_data_results(
    num_samples: int = 3,
    fig_size=(20, 10),
    *,
    predict_config: PredictConfig,
    img_ch: int = 3,
    n_bands: int = 3
):
    """
    Plots some samples from the results directory.
    Parameters
    ----------
    num_samples : int
        Number of samples to plot.
    fig_size : tuple
        Figure size.
    img_ch : int
        Number of channels.
    predict_config : PredictConfig
        Prediction onfiguration object.
    """

    images = [
        os.path.basename(f)
        for f in sorted(glob(os.path.join(predict_config.results_path, "*.tif")))
    ]

    images = random.sample(images, num_samples)
    for img_file in images:
        try:
            if n_bands not in (1, 3):
                raise RuntimeError("n_bands option must be 1 or 3")

            fig, axes = plt.subplots(
                nrows=1, ncols=predict_config.n_classes + 1, figsize=(20, 40)
            )

            if n_bands == 1:

                img_s2 = tiff.imread(
                    os.path.join(predict_config.images_path, "images", img_file)
                )[:, :, img_ch]
                axes[0].imshow(img_s2)

            if n_bands == 3:

                img_s2 = tiff.imread(
                    os.path.join(predict_config.images_path, "images", img_file)
                )[:, :, :3]
                axes[0].imshow(img_s2)

            # Prediccion
            mask_ = (
                tiff.imread(os.path.join(predict_config.results_path, img_file)) / 255
            )

            mask_ = resize(
                mask_,
                (predict_config.height, predict_config.width, predict_config.n_classes),
                mode="constant",
                preserve_range=True,
            )

            for c in range(predict_config.n_classes):
                axes[1 + c].imshow(np.squeeze(mask_[:, :, c]))

            plt.show()

        except Exception as err:
            print(err)

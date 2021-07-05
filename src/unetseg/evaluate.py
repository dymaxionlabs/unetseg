import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from sklearn.preprocessing import minmax_scale

from unetseg.train import build_data_generator


def plot_data_generator(num_samples=3, fig_size=(20, 10), *, train_config, img_ch=3):

    if train_config.n_channels < 4:
        img_ch = train_config.n_channels
    else:
        img_ch = img_ch

    images_dir = os.path.join(train_config.images_path, "images")
    mask_dir = os.path.join(train_config.images_path, "masks")

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

                image = image[:, :, :img_ch].astype(np.uint8)
                ax[0].imshow(image)
                for i in range(train_config.n_classes):
                    ax[i + 1].imshow(mask[:, :, i])
                j += 1
                if j >= num:
                    return

    plot_samples(plt, data_generator, num_samples)
    plt.show()


def plot_data_results(num_samples=3, fig_size=(20, 10), *, predict_config, img_ch=3):

    if predict_config.n_channels < 4:
        img_ch = predict_config.n_channels
    else:
        img_ch = img_ch

    images = [
        os.path.basename(f)
        for f in sorted(glob(os.path.join(predict_config.results_path, "*.tif")))
    ]

    images = random.sample(images, num_samples)
    for img_file in images:
        try:
            # s2 3D
            img_s2 = tiff.imread(
                os.path.join(predict_config.images_path, "images", img_file)
            )[:, :, :img_ch]
            img_s2 = minmax_scale(img_s2.ravel(), feature_range=(0, 255)).reshape(
                img_s2.shape
            )

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

            fig, axes = plt.subplots(
                nrows=1, ncols=predict_config.n_classes + 1, figsize=(20, 40)
            )

            img_s2 = img_s2.astype(np.uint8)

            axes[0].imshow(img_s2)
            for c in range(predict_config.n_classes):
                axes[1 + c].imshow(np.squeeze(mask_[:, :, c]))

            plt.show()

        except Exception as err:
            print(err)

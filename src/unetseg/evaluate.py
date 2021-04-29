import os
from glob import glob

import matplotlib.pyplot as plt

from unetseg.train import build_data_generator


def plot_data_generator(num_samples=3, fig_size=(20, 10), *, train_config):
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
                ax[0].imshow(image)
                for i in range(train_config.n_classes):
                    ax[i + 1].imshow(mask[:, :, i])
                j += 1
                if j >= num:
                    return

    plot_samples(plt, data_generator, num_samples)
    plt.show()

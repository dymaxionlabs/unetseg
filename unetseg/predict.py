import os
import warnings
from glob import glob

import attr
import numpy as np
import rasterio
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from unetseg.train import build_model_unet, build_model_unetplusplus
from unetseg.utils import grouper, resize

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")


@attr.s
class PredictConfig:
    images_path = attr.ib(default="")
    results_path = attr.ib(default="")
    batch_size = attr.ib(default=32)
    model_architecture = attr.ib(default="unet")
    model_path = attr.ib(default="unet.h5")
    height = attr.ib(default=320)
    width = attr.ib(default=320)
    n_channels = attr.ib(default=3)
    n_classes = attr.ib(default=1)
    class_weights = attr.ib(default=0)


def predict(cfg: PredictConfig):
    """
    Performs inference based on a configuration object

    Parameters
    ----------
    cfg : PredictConfig
        Configuration object

    """
    pat = os.path.join(cfg.images_path, "images/*.tif")
    predict_ids = glob(pat)
    print(f"Total images to predict ({pat}):", len(predict_ids))

    os.makedirs(cfg.results_path, exist_ok=True)

    # Skip already existing files
    predict_ids = [
        p
        for p in predict_ids
        if not os.path.exists(os.path.join(cfg.results_path, os.path.basename(p)))
    ]
    print("After skipping existing results:", len(predict_ids))

    # Load model
    # FIXME: Find a better way to load model (.load_model() did not work because
    # of the weighted_binary_crossentropy function).
    # build_model() only expects cfg to have: width, height, n_channels, n_classes

    if cfg.model_architecture == "unet":
        model = build_model_unet(cfg)
        # print(model.summary())
    elif cfg.model_architecture == "unetplusplus":
        model = build_model_unetplusplus(cfg)
        # print(model.summary())
    else:
        print("no model architecture was set so default UNet model will be use")
        model = build_model_unet(cfg)
        # print(model.summary())

    model.load_weights(cfg.model_path)

    # Predict over each batch of images
    groups = list(grouper(predict_ids, cfg.batch_size))
    for mini_group in tqdm(groups):
        mini_group = [g for g in mini_group if g]

        X_predict = []
        X_profile = []

        for i, img_path in enumerate(mini_group):
            with rasterio.open(img_path) as src:
                img_ = np.dstack([src.read(b) for b in range(1, cfg.n_channels + 1)])
                profile_ = src.profile.copy()
                img_ = minmax_scale(img_.ravel(), feature_range=(0, 255)).reshape(
                    img_.shape
                )
                img_ = resize(img_, (cfg.height, cfg.width))
                img_ = img_.reshape(cfg.height, cfg.width, cfg.n_channels)
                X_predict.append(img_)
                X_profile.append(profile_)

        # Predict batch
        pred = model.predict(np.array(X_predict))
        preds_test_ = pred  # > 0.05

        preds_test_scaled_ = minmax_scale(
            preds_test_.ravel(), feature_range=(1, 255)
        ).reshape(preds_test_.shape)

        for i, img_path in enumerate(mini_group):
            profile_ = X_profile[i]
            profile_.update(count=cfg.n_classes, dtype=np.uint8)

            out_height, out_width = profile_["height"], profile_["width"]

            filename = os.path.basename(img_path)
            with rasterio.open(
                os.path.join(cfg.results_path, filename), "w", **profile_
            ) as dst:

                img = resize(preds_test_scaled_[i], (out_height, out_width))
                img = img.astype(np.uint8).reshape(
                    (out_height, out_width, cfg.n_classes)
                )
                for b in range(cfg.n_classes):
                    dst.write(img[:, :, b], b + 1)

    print("Done!")

import os
import random
import warnings
from glob import glob
from typing import List, Tuple

import albumentations as A
import attr
import numpy as np
import rasterio
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model

from unetseg.utils import resize

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")


@attr.s
class TrainConfig:
    images_path = attr.ib()
    masks_path = attr.ib(default=None)
    width = attr.ib(default=200)
    height = attr.ib(default=200)
    n_channels = attr.ib(default=3)
    n_classes = attr.ib(default=1)
    apply_image_augmentation = attr.ib(default=True)
    model_path = attr.ib(default="unet.h5")
    model_architecture = attr.ib(default="unet")
    validation_split = attr.ib(default=0.1)
    test_split = attr.ib(default=0.1)
    epochs = attr.ib(default=15)
    steps_per_epoch = attr.ib(default=2000)
    early_stopping_patience = attr.ib(default=3)
    batch_size = attr.ib(default=32)
    seed = attr.ib(default=None)
    evaluate = attr.ib(default=True)
    class_weights = attr.ib(default=0)


def build_model_unetplusplus(cfg: TrainConfig) -> Model:
    """
    Builds a U-Net++ model.

    Parameters
    ----------
    cfg : TrainConfig
        Training configuration.

    Returns
    -------
    Model
        The U-Net++ model.

    """
    # NOTE: for now, classes are equally balanced
    if cfg.class_weights == 0:
        cfg.class_weights = [0.5 for _ in range(cfg.n_classes)]

    # growth_factor = 2
    # droprate = 0.25
    number_of_filters = 2
    # upconv = True
    # batch_size = cfg.batch_size

    def conv2d(filters: int):
        return Conv2D(filters=filters, kernel_size=(3, 3), padding="same")

    def conv2dtranspose(filters: int):
        return Conv2DTranspose(
            filters=filters, kernel_size=(2, 2), strides=(2, 2), padding="same"
        )

    model_input = Input((cfg.height, cfg.width, cfg.n_channels))
    x00 = conv2d(filters=int(16 * number_of_filters))(model_input)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(0.2)(x00)
    x00 = conv2d(filters=int(16 * number_of_filters))(x00)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(0.2)(x00)
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)

    x10 = conv2d(filters=int(32 * number_of_filters))(p0)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(0.2)(x10)
    x10 = conv2d(filters=int(32 * number_of_filters))(x10)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(0.2)(x10)
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)

    x01 = conv2dtranspose(int(16 * number_of_filters))(x10)
    x01 = concatenate([x00, x01])
    x01 = conv2d(filters=int(16 * number_of_filters))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = conv2d(filters=int(16 * number_of_filters))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = Dropout(0.2)(x01)

    x20 = conv2d(filters=int(64 * number_of_filters))(p1)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(0.2)(x20)
    x20 = conv2d(filters=int(64 * number_of_filters))(x20)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(0.2)(x20)
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)

    x11 = conv2dtranspose(int(16 * number_of_filters))(x20)
    x11 = concatenate([x10, x11])
    x11 = conv2d(filters=int(16 * number_of_filters))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = conv2d(filters=int(16 * number_of_filters))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = Dropout(0.2)(x11)

    x02 = conv2dtranspose(int(16 * number_of_filters))(x11)
    x02 = concatenate([x00, x01, x02])
    x02 = conv2d(filters=int(16 * number_of_filters))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = conv2d(filters=int(16 * number_of_filters))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = Dropout(0.2)(x02)

    x30 = conv2d(filters=int(128 * number_of_filters))(p2)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(0.2)(x30)
    x30 = conv2d(filters=int(128 * number_of_filters))(x30)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(0.2)(x30)
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)

    x21 = conv2dtranspose(int(16 * number_of_filters))(x30)
    x21 = concatenate([x20, x21])
    x21 = conv2d(filters=int(16 * number_of_filters))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = conv2d(filters=int(16 * number_of_filters))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = Dropout(0.2)(x21)

    x12 = conv2dtranspose(int(16 * number_of_filters))(x21)
    x12 = concatenate([x10, x11, x12])
    x12 = conv2d(filters=int(16 * number_of_filters))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = conv2d(filters=int(16 * number_of_filters))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = Dropout(0.2)(x12)

    x03 = conv2dtranspose(int(16 * number_of_filters))(x12)
    x03 = concatenate([x00, x01, x02, x03])
    x03 = conv2d(filters=int(16 * number_of_filters))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = conv2d(filters=int(16 * number_of_filters))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = Dropout(0.2)(x03)

    m = conv2d(filters=int(256 * number_of_filters))(p3)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = conv2d(filters=int(256 * number_of_filters))(m)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = Dropout(0.2)(m)

    x31 = conv2dtranspose(int(128 * number_of_filters))(m)
    x31 = concatenate([x31, x30])
    x31 = conv2d(filters=int(128 * number_of_filters))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = conv2d(filters=int(128 * number_of_filters))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = Dropout(0.2)(x31)

    x22 = conv2dtranspose(int(64 * number_of_filters))(x31)
    x22 = concatenate([x22, x20, x21])
    x22 = conv2d(filters=int(64 * number_of_filters))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = conv2d(filters=int(64 * number_of_filters))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = Dropout(0.2)(x22)

    x13 = conv2dtranspose(int(32 * number_of_filters))(x22)
    x13 = concatenate([x13, x10, x11, x12])
    x13 = conv2d(filters=int(32 * number_of_filters))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = conv2d(filters=int(32 * number_of_filters))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = Dropout(0.2)(x13)

    x04 = conv2dtranspose(int(16 * number_of_filters))(x13)
    x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
    x04 = conv2d(filters=int(16 * number_of_filters))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = conv2d(filters=int(16 * number_of_filters))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = Dropout(0.2)(x04)

    output = Conv2D(cfg.n_classes, kernel_size=(1, 1), activation="sigmoid")(x04)

    model = Model(inputs=[model_input], outputs=[output])

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(cfg.class_weights))

    model.compile(
        optimizer="adam",
        # optimizer=Adam(),#tf.optimizers.Adam(lr=0.0005),
        loss=weighted_binary_crossentropy,
        metrics=[tf.keras.metrics.MeanIoU(num_classes=cfg.n_classes + 1)],
    )

    return model


def build_model_unet(cfg: TrainConfig) -> Model:
    """
    Build U-Net model class.

    Parameters
    ----------
    cfg : TrainConfig
        Configuration for training.

    Returns
    -------
    Model
        U-Net model class.

    """
    # NOTE: for now, classes are equally balanced
    if cfg.class_weights == 0:
        cfg.class_weights = [0.5 for _ in range(cfg.n_classes)]

    growth_factor = 2
    n_filters_start = 32
    droprate = 0.25
    n_filters = n_filters_start
    upconv = True
    inputs = Input((cfg.height, cfg.width, cfg.n_channels))
    # inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_1 = Dropout(droprate)(pool4_1)

    n_filters *= growth_factor
    pool4_1 = BatchNormalization()(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    pool4_2 = Dropout(droprate)(pool4_2)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(pool4_2)
    conv5 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv5)

    n_filters //= growth_factor
    if upconv:
        up6_1 = concatenate(
            [
                Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding="same")(
                    conv5
                ),
                conv4_1,
            ]
        )
    else:
        up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
    up6_1 = BatchNormalization()(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv6_1)
    conv6_1 = Dropout(droprate)(conv6_1)

    n_filters //= growth_factor
    if upconv:
        up6_2 = concatenate(
            [
                Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding="same")(
                    conv6_1
                ),
                conv4_0,
            ]
        )
    else:
        up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv6_2)
    conv6_2 = Dropout(droprate)(conv6_2)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate(
            [
                Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding="same")(
                    conv6_2
                ),
                conv3,
            ]
        )
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv7)
    conv7 = Dropout(droprate)(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate(
            [
                Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding="same")(
                    conv7
                ),
                conv2,
            ]
        )
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv8)
    conv8 = Dropout(droprate)(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate(
            [
                Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding="same")(
                    conv8
                ),
                conv1,
            ]
        )
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(cfg.n_classes, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(cfg.class_weights))

    model.compile(
        optimizer="adam",
        # optimizer=Adam(),#tf.optimizers.Adam(),
        loss=weighted_binary_crossentropy,
        metrics=[tf.keras.metrics.MeanIoU(num_classes=cfg.n_classes + 1)],
    )

    return model


def preprocess_input(
    image: np.ndarray, mask: np.ndarray, *, config: TrainConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess input image and masks.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    mask : np.ndarray
        Input mask.
    config : TrainConfig
        Training configuration.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Preprocessed image and mask.

    """
    # Scale image to 0-255 range
    image_ = minmax_scale(image.ravel(), feature_range=(0, 255)).reshape(image.shape)
    # Scale to 0-1 by dividing by 255 (we assume that mask has true values
    # filled with 255, and false values as 0).
    mask_ = mask / 255
    size = (config.height, config.width)

    # Resize image and mask
    image_ = resize(image_, size)
    mask_ = resize(mask_, size)

    if config.apply_image_augmentation:
        # Add extra augmentations to image and mask
        aug_pipeline = A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
        res = aug_pipeline(image=image_.astype(np.uint8), mask=mask_)
        image_, mask_ = res["image"], res["mask"]

    # In case mask is binary (1 class),
    # make sure mask has shape (H, W, 1) and not (H, W).
    image_ = image_.reshape(image_.shape[0], image_.shape[1], config.n_channels)
    mask_ = mask_.reshape(mask_.shape[0], mask_.shape[1], config.n_classes)

    return image_, mask_


def get_raster(image_path: str, n_channels: int = None) -> np.ndarray:
    """
    Loads a raster image from a file.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    n_channels : int, optional
        Number of channels in the image. If not specified, the number of channels
        is inferred from the image file.

    Returns
    -------
    np.ndarray
        The loaded image.

    """
    with rasterio.open(image_path) as src:
        if not n_channels:
            n_channels = src.count
        return np.dstack([src.read(b) for b in range(1, n_channels + 1)])


def get_mask_raster(
    image_path: str, n_channels: int = None, *, mask_dir: str
) -> np.ndarray:
    """
    Get respective mask raster from image path.

    Parameters
    ----------
    image_path : str
        Path to image.
    n_channels : int, optional
        Number of channels in image. The default is None.
    mask_dir : str, optional
        Path to mask directory. The default is None.

    Returns
    -------
    np.ndarray
        Mask image.

    """
    basename = os.path.basename(image_path)
    mask_path = os.path.join(mask_dir, basename)
    with rasterio.open(mask_path) as src:
        if not n_channels:
            n_channels = src.count
        return np.dstack([src.read(b) for b in range(1, n_channels + 1)])


def build_data_generator(image_files: List[str], *, config: TrainConfig, mask_dir: str):
    """
    Build data generator based on a list of images and directory of binary masks.

    Parameters
    ----------
    image_files : List[str]
        List of paths to images.
    config : TrainConfig
        Configuration object.
    mask_dir : str
        Path to directory with binary masks.

    Yields
    ------
    tuple
        Tuple of image and mask batch.

    """
    if not image_files:
        raise RuntimeError("image_files is empty")

    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=image_files, size=config.batch_size)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = get_raster(input_path, n_channels=config.n_channels)
            mask = get_mask_raster(
                input_path, mask_dir=mask_dir, n_channels=config.n_classes
            )

            input, mask = preprocess_input(image=input, mask=mask, config=config)

            if not np.any(np.isnan(input)):
                if not np.any(np.isnan(mask)):

                    batch_input.append(input)
                    batch_output.append(mask)

        # Return a tuple of (input, output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield batch_x, batch_y


def train(cfg: TrainConfig):
    """
    Performs training and evaluation of the model based on a configuration object.

    Parameters
    ----------
    cfg : TrainConfig
        Configuration object containing all the necessary parameters for training.

    """
    if cfg.seed:
        random.seed = cfg.seed
        np.random.seed = cfg.seed

    if cfg.model_architecture == "unet":
        model = build_model_unet(cfg)
        print(model.summary())
    elif cfg.model_architecture == "unetplusplus":
        model = build_model_unetplusplus(cfg)
        print(model.summary())
    else:
        print("no model architecture was set so default UNet model will be use")
        model = build_model_unet(cfg)
        print(model.summary())

    all_images = glob(os.path.join(cfg.images_path, "images", "*.tif"))
    print("All images:", len(all_images))

    # Split dataset by shuffling and taking the first N elements for validation,
    # and the rest for training.
    np.random.shuffle(all_images)

    n_test = round(cfg.test_split * len(all_images))
    n_val = round(cfg.validation_split * (len(all_images) - n_test))

    test_images, all_train_images = all_images[:n_test], all_images[n_test:]
    val_images, train_images = all_train_images[:n_val], all_train_images[n_val:]

    print("Num. training images:", len(train_images))
    print("Num. validation images:", len(val_images))
    print("Num. test images:", len(test_images))

    if not train_images:
        raise RuntimeError("train_images is empty")
    if not val_images:
        raise RuntimeError("val_images is empty")
    if not test_images:
        raise RuntimeError("test_images is empty")

    mask_dir = cfg.masks_path or os.path.join(cfg.images_path, "extent")
    train_generator = build_data_generator(train_images, config=cfg, mask_dir=mask_dir)
    val_generator = build_data_generator(val_images, config=cfg, mask_dir=mask_dir)
    test_generator = build_data_generator(test_images, config=cfg, mask_dir=mask_dir)

    # Make sure weights dir exist
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)

    print("Compile and fit the UNet model")
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience, verbose=1)
    checkpoint = ModelCheckpoint(cfg.model_path, verbose=1, save_best_only=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                               patience=5, min_lr=0.001)

    results = model.fit(
        train_generator,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_data=val_generator,
        validation_steps=round(cfg.steps_per_epoch * cfg.validation_split),
        callbacks=[early_stopping, checkpoint],
    )

    # Save model
    model.save(cfg.model_path)

    # Evaluate model on test set
    if cfg.evaluate:
        scores = model.evaluate(
            test_generator, steps=len(test_images) // cfg.batch_size
        )
        loss, mean_iou = scores
        print("*** Final  metrics ***")
        print("Loss:", loss)
        print("Mean IoU:", mean_iou)

    return results

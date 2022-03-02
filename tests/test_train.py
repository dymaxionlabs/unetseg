import tempfile
from pathlib import Path

from unetseg.train import TrainConfig, train

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "apache-2.0"


def test_train_unet(shared_datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        images_path = shared_datadir / "train"
        model_path = Path(tmpdir) / "unet_model.h5"

        cfg = TrainConfig(
            images_path=images_path,
            width=160,
            height=160,
            n_channels=3,
            n_classes=1,
            apply_image_augmentation=True,
            seed=42,
            epochs=1,
            steps_per_epoch=10,
            batch_size=4,
            model_architecture="unet",
            evaluate=True,
            class_weights=[1],
            model_path=model_path,
        )

        res = train(cfg)

        assert res.params["epochs"] == 1
        assert res.params["steps"] == 10
        assert len(res.epoch) == 1
        assert model_path.is_file()


def test_train_unetplusplus(shared_datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        images_path = shared_datadir / "train"
        model_path = Path(tmpdir) / "unet_model.h5"

        cfg = TrainConfig(
            images_path=images_path,
            width=160,
            height=160,
            n_channels=3,
            n_classes=1,
            apply_image_augmentation=True,
            seed=42,
            epochs=1,
            steps_per_epoch=10,
            batch_size=4,
            model_architecture="unetplusplus",
            evaluate=True,
            class_weights=[1],
            model_path=model_path,
        )

        res = train(cfg)

        assert res.params["epochs"] == 1
        assert res.params["steps"] == 10
        assert len(res.epoch) == 1
        assert model_path.is_file()

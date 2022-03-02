import tempfile
from pathlib import Path

from unetseg.predict import PredictConfig, predict
from unetseg.train import TrainConfig, train

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "apache-2.0"


def train_test_model(*, images_path, model_path):
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
    train(cfg)


def test_predict(shared_datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        images_path = shared_datadir / "train"
        model_path = Path(tmpdir) / "unet_model.h5"
        results_path = Path(tmpdir) / "results"

        # Train a test model first, to predict with
        train_test_model(images_path=images_path, model_path=model_path)
        assert model_path.is_file()

        cfg = PredictConfig(
            images_path=images_path,
            results_path=results_path,
            batch_size=4,
            model_path=model_path,
            height=160,
            width=160,
            n_channels=3,
            n_classes=1,
            class_weights=[1],
        )
        predict(cfg)

        image_files = list((images_path / "images").glob("*.tif"))
        result_files = list(results_path.glob("*.tif"))
        assert len(result_files) == len(image_files)

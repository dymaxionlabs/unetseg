import matplotlib

from unetseg.evaluate import plot_data_generator, plot_data_results
from unetseg.predict import PredictConfig
from unetseg.train import TrainConfig

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "apache-2.0"


def test_plot_data_generator(shared_datadir):
    matplotlib.use("agg")

    images_path = shared_datadir / "train"
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
    )

    plot_data_generator(num_samples=4, train_config=cfg)


def test_plot_data_results(shared_datadir):
    matplotlib.use("agg")

    images_path = shared_datadir / "train"
    results_path = shared_datadir / "results"

    cfg = PredictConfig(
        images_path=images_path,
        results_path=results_path,
        batch_size=4,
        height=160,
        width=160,
        n_channels=3,
        n_classes=1,
        class_weights=[1],
    )

    plot_data_results(predict_config=cfg)

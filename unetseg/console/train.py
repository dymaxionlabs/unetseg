# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = unetseg.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import logging
import sys

from unetseg import __version__
from unetseg.train import TrainConfig, train

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "apache-2.0"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Train a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="unetseg {ver}".format(ver=__version__)
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    parser.add_argument(
        "train_dir",
        help="Path to image tiles and masks (directory with images/ and masks/)",
    )
    parser.add_argument(
        "-o", "--output", help="path to output model (.h5)", default="./unet.h5"
    )
    parser.add_argument("-W", "--width", type=int, help="Image tile width")
    parser.add_argument("-H", "--height", type=int, help="Image tile height")
    parser.add_argument(
        "-N", "--num-channels", default=3, type=int, help="Number of channels"
    )
    parser.add_argument(
        "-C", "--num-classes", default=1, type=int, help="Number of classes"
    )
    parser.add_argument(
        "-E", "--epochs", default=15, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--steps-per-epoch", default=100, type=int, help="steps per epoch"
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=3,
        type=int,
        help="number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="batch size")
    parser.add_argument(
        "--image-augmentation",
        dest="image_augmentation",
        help="Apply image augmentation",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-image-augmentation",
        dest="image_augmentation",
        help="Do not apply image augmentation",
        action="store_false",
        default=False,
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        help="Evaluate metrics over validation set at the end of training",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        help="Do not evaluate emtrics over validation set at the end of training",
        action="store_false",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        type=int,
        help="Seed number for the random number generation",
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    config = TrainConfig(
        width=args.width,
        height=args.height,
        n_channels=args.num_channels,
        n_classes=args.num_classes,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        early_stopping_patience=args.early_stopping_patience,
        apply_image_augmentation=args.image_augmentation,
        batch_size=args.batch_size,
        seed=args.seed,
        images_path=args.train_dir,
        evaluate=args.evaluate,
        model_path=args.output,
    )

    train(config)


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

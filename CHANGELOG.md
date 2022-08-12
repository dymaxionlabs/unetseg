# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Save Tensorboard and CSV logs when training

## [0.2.1] - 2022-02-02

### Changed

- fix(evaluate): Change `masks` to `extent`

## [0.2.0] - 2022-01-13

### Added

- New `masks_path` training config attribute for specifying a custom directory for masks

### Changed

- Upgrade to Tensorflow 2+
- Support for satproc >= 0.1.8
- Depend on Python >= 3.7

## [0.1.10] - 2022-01-11

### Changed

- Force requirement to Python < 3.8 (caused by dependency to TF 1.15)
- Add missing dependencies
- Update docstrings

## [0.1.5] - 2021-08-18

### Added

- New model Unet++ and setting `model_architecture` on training config object

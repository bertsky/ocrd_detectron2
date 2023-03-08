# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- avoid colon in generated region IDs

## [0.1.5] - 2023-01-15
### Fixed
- param `debug_img`: 1 image per page
- URLs/specs for PubLayNet/JPLeoRX models

## [0.1.4] - 2022-12-02
### Added
- param `postprocessing` (select steps, including `none`)
- param `debug_img` (styles to visualise raw predictions, including `none`)

## [0.1.3] - 2022-11-02
### Fixed
- `make deps`: fall back to Detectron2 src build

### Changed
- added various models as file resources
- added corresponding preset files
- updated documentation

## [0.1.2] - 2022-10-27
### Fixed
- `make deps`: fix CUDA detection even more
- apply `device` param as passed

### Changed
- downscale images to no more than 150 DPI for prediction (for speed)
- add param `operation_level` (default `page`), add `table` mode

## [0.1.1] - 2022-02-02
### Fixed
- `make deps`: fix CUDA detection and allow CPU as fallback

### Changed
- instance segmentation postprocessing: use asymmetric overlap
  criterion for non-maximum suppression
- skip instances which belong to classes with empty category
- annotate incrementally (by skipping candidates that overlap
  with pre-existing top-level regions)

## [0.1.0] - 2022-01-21

<!-- link-labels -->
[0.1.0]: ../../compare/aeca7e37...v0.1.0
[0.1.1]: ../../compare/v0.1.0...v0.1.1
[0.1.2]: ../../compare/v0.1.1...v0.1.2
[0.1.3]: ../../compare/v0.1.2...v0.1.3
[0.1.4]: ../../compare/v0.1.3...v0.1.4
[0.1.5]: ../../compare/v0.1.4...v0.1.5
[unreleased]: ../../compare/v0.1.4...master

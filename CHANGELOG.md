# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8] - 2023-06-29
### Fixed
- workarounds for broken models (DocBank_X101, Jambo-sudo_X101)
- `make deps`: add explicit reqs prior to pip step with Torch index
- set `pc:PcGts/@pcGtsId` from `mets:file/@ID`

### Added
- CI for CLI tests (with cached models and stored result artifacts)

### Changed
- migrated model URLs from external to Github release assets

## [0.1.7] - 2023-03-20
### Fixed
- adapt to Numpy 1.24 (no `np.bool`)

### Added
- model by Jambo-sudo (PubLayNet+custom GT)
- model by LayoutParser (PRImA Layout GT)
- CLI tests

## [0.1.6] - 2023-03-10
### Fixed
- avoid colon in generated region IDs
- `make deps`: add explicit deps for torch
- fix/update file resources
- fix model config base paths on-the-fly

### Added
- add Psarpei TD model

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
[0.1.6]: ../../compare/v0.1.5...v0.1.6
[0.1.7]: ../../compare/v0.1.6...v0.1.7
[0.1.8]: ../../compare/v0.1.7...v0.1.8
[unreleased]: ../../compare/v0.1.8...master

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- `make deps`: fix CUDA detection even more
- apply `device` param as passed

### Changed
- downscale images to no more than 150 DPI for prediction (for speed)

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
[unreleased]: ../../compare/v0.1.1...master
[0.1.1]: ../../compare/v0.1.0...v0.1.1

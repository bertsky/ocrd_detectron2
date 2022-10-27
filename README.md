[![PyPI version](https://badge.fury.io/py/ocrd-detectron2.svg)](https://badge.fury.io/py/ocrd-detectron2)

# ocrd_detectron2

    OCR-D wrapper for detectron2 based segmentation models

  * [Introduction](#introduction)
  * [Installation](#installation)
  * [Usage](#usage)
     * [OCR-D processor interface ocrd-detectron2-segment](#ocr-d-processor-interface-ocrd-detectron2-segment)
  * [Models](#models)
     * [TableBank](#tablebank)
     * [PubLayNet](#publaynet)
     * [PubLayNet](#publaynet-1)
     * [LayoutParser](#layoutparser)
     * [DocBank](#docbank)
  * [Testing](#testing)

## Introduction

This offers [OCR-D](https://ocr-d.de) compliant [workspace processors](https://ocr-d.de/en/spec/cli) for document layout analysis with models trained on [Detectron2](https://github.com/facebookresearch/detectron2), which implements [Faster R-CNN](https://arxiv.org/abs/1506.01497), [Mask R-CNN](https://arxiv.org/abs/1703.06870), [Cascade R-CNN](https://arxiv.org/abs/1712.00726), [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) and [Panoptic Segmentation](https://arxiv.org/abs/1801.00868), among others.

In trying to cover a broad range of third-party models, a few sacrifices have to be made: Deployment of [models](#models) may be difficult, and needs configuration. Class labels (really [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML) region types) must be provided. The code itself tries to cope with panoptic and instance segmentation models (with or without masks).

Only meant for (coarse) page segmentation into regions – no text lines, no reading order, no orientation.

## Installation

Create and activate a [virtual environment](https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments) as usual.

To install Python dependencies:

    make deps

Which is the equivalent of:

    pip install -r requirements.txt -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html # for CUDA 11.3
    pip install -r requirements.txt -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html # for CPU only

To install this module, then do:

    make install

Which is the equivalent of:

    pip install .

## Usage

### [OCR-D processor](https://ocr-d.de/en/spec/cli) interface `ocrd-detectron2-segment`

To be used with [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML) documents in an [OCR-D](https://ocr-d.de/en/about) annotation workflow.

```
Usage: ocrd-detectron2-segment [OPTIONS]

  Detect regions with Detectron2 models

  > Use detectron2 to segment each page into regions.

  > Open and deserialize PAGE input files and their respective images.
  > Fetch a raw and a binarized image for the page frame (possibly
  > cropped and deskewed).

  > Feed the raw image into the detectron2 predictor that has been used
  > to load the given model. Then, depending on the model capabilities
  > (whether it can do panoptic segmentation or only instance
  > segmentation, whether the latter can do masks or only bounding
  > boxes), post-process the predictions:

  > - panoptic segmentation: take the provided segment label map, and
  >   apply the segment to class label map,
  > - instance segmentation: find an optimal non-overlapping set (flat
  >   map) of instances via non-maximum suppression,
  > - both: avoid overlapping pre-existing top-level regions (incremental
  >   segmentation).

  > Then extend / shrink the surviving masks to fully include / exclude
  > connected components in the foreground that are on the boundary.

  > Finally, find the convex hull polygon for each region, and map its
  > class id to a new PAGE region type (and subtype).

  > (Does not annotate `ReadingOrder` or `TextLine`s or `@orientation`.)

  > Produce a new output file by serialising the resulting hierarchy.

Options:
  -I, --input-file-grp USE        File group(s) used as input
  -O, --output-file-grp USE       File group(s) used as output
  -g, --page-id ID                Physical page ID(s) to process
  --overwrite                     Remove existing output pages/images
                                  (with --page-id, remove only those)
  --profile                       Enable profiling
  --profile-file                  Write cProfile stats to this file. Implies --profile
  -p, --parameter JSON-PATH       Parameters, either verbatim JSON string
                                  or JSON file path
  -P, --param-override KEY VAL    Override a single JSON object key-value pair,
                                  taking precedence over --parameter
  -m, --mets URL-PATH             URL or file path of METS to process
  -w, --working-dir PATH          Working directory of local workspace
  -l, --log-level [OFF|ERROR|WARN|INFO|DEBUG|TRACE]
                                  Log level
  -C, --show-resource RESNAME     Dump the content of processor resource RESNAME
  -L, --list-resources            List names of processor resources
  -J, --dump-json                 Dump tool description as JSON and exit
  -D, --dump-module-dir           Output the 'module' directory with resources for this processor
  -h, --help                      This help message
  -V, --version                   Show version

Parameters:
   "categories" [array - REQUIRED]
    maps each region category (position) of the model to a PAGE region
    type (and @type or @custom if separated by colon), e.g.
    ['TextRegion:paragraph', 'TextRegion:heading',
    'TextRegion:floating', 'TableRegion', 'ImageRegion'] for PubLayNet;
    categories with an empty string will be skipped during prediction
   "min_confidence" [number - 0.5]
    confidence threshold for detections
   "model_config" [string - REQUIRED]
    path name of model config
   "model_weights" [string - REQUIRED]
    path name of model weights
   "device" [string - "cuda"]
    select computing device for Torch (e.g. cpu or cuda:0); will fall
    back to CPU if no GPU is available
```

Example:

    ocrd resmgr download ocrd-detectron2-segment TableBank_X152.yaml
    ocrd resmgr download ocrd-detectron2-segment TableBank_X152.pth
    ocrd-detectron2-segment -I OCR-D-BIN -O OCR-D-SEG-TAB -P categories '["TableRegion"]' -P model_config TableBank_X152.yaml -P model_weights TableBank_X152.pth -P min_confidence 0.1
    ocrd-detectron2-segment -I OCR-D-BIN -O OCR-D-SEG-TAB -p presets_TableBank_X152.json -P min_confidence 0.1 # equivalent, with presets file
    ocrd resmgr download ocrd-detectron2-segment "*" # get all preconfigured models

## Models

Some of the following models have already been registered as known [file resources](https://ocr-d.de/en/spec/cli#processor-resources), along with parameter presets to use them.

To get a list of available registered models, do:

    ocrd resmgr list-available -e ocrd-detectron2-segment

To get a list of already installed models and presets, do:

    ocrd resmgr list-installed -e ocrd-detectron2-segment

To download a registered model (i.e. a config file and the respective weights file), do:

    ocrd resmgr download ocrd-detectron2-segment NAME.yaml
    ocrd resmgr download ocrd-detectron2-segment NAME.pth

To download more models (registered or other), see:

    ocrd resmgr download --help

To use a model, do:

    ocrd-detectron2-segment -P model_config NAME.yaml -P model_weights NAME.pth -P categories '[...]' ...
    ocrd-detectron2-segment -p NAME.json ... # equivalent, with presets file

> **Note**: These are just examples, no exhaustive search was done yet!

> **Note**: Make sure you unpack first if the download link is an archive. Also, the filename suffix (.pth vs .pkl) of the weight file does matter!

### [TableBank](https://github.com/doc-analysis/TableBank)

R152-FPN [config](https://layoutlm.blob.core.windows.net/tablebank/model_zoo/detection/All_X152/All_X152.yaml)|[weights](https://layoutlm.blob.core.windows.net/tablebank/model_zoo/detection/All_X152/model_final.pth)|`["TableRegion"]`

### [PubLayNet](https://github.com/hpanwar08/detectron2)

R50-FPN [config](https://github.com/hpanwar08/detectron2/raw/master/configs/DLA_mask_rcnn_R_50_FPN_3x.yaml)|[weights](https://www.dropbox.com/sh/44ez171b2qaocd2/AAB0huidzzOXeo99QdplZRjua)|`["TextRegion:paragraph", "TextRegion:heading", "TextRegion:floating", "TableRegion", "ImageRegion"]`

R101-FPN [config](https://github.com/hpanwar08/detectron2/raw/master/configs/DLA_mask_rcnn_R_101_FPN_3x.yaml)|[weights](https://www.dropbox.com/sh/wgt9skz67usliei/AAD9n6qbsyMz1Y3CwpZpHXCpa)|`["TextRegion:paragraph", "TextRegion:heading", "TextRegion:floating", "TableRegion", "ImageRegion"]`

X101-FPN [config](https://github.com/hpanwar08/detectron2/raw/master/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml)|[weights](https://www.dropbox.com/sh/1098ym6vhad4zi6/AABe16eSdY_34KGp52W0ruwha)|`["TextRegion:paragraph", "TextRegion:heading", "TextRegion:floating", "TableRegion", "ImageRegion"]`

### [PubLayNet](https://github.com/JPLeoRX/detectron2-publaynet)

R50-FPN [config](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml)|[weights](https://keybase.pub/jpleorx/detectron2-publaynet/mask_rcnn_R_50_FPN_3x/model_final.pth)|`["TextRegion:paragraph", "TextRegion:heading", "TextRegion:floating", "TableRegion", "ImageRegion"]`

R101-FPN [config](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml)|[weights](https://keybase.pub/jpleorx/detectron2-publaynet/mask_rcnn_R_101_FPN_3x/model_final.pth)|`["TextRegion:paragraph", "TextRegion:heading", "TextRegion:floating", "TableRegion", "ImageRegion"]`

### [LayoutParser](https://github.com/Layout-Parser/layout-parser/blob/master/src/layoutparser/models/detectron2/catalog.py)

provides different model variants of various depths for multiple datasets:
- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) (Medical Research Papers)
- [TableBank](https://doc-analysis.github.io/tablebank-page/index.html) (Tables Computer Typesetting)
- [PRImALayout](https://www.primaresearch.org/dataset/) (Various Computer Typesetting)
- [HJDataset](https://dell-research-harvard.github.io/HJDataset/) (Historical Japanese Magazines)
- [NewspaperNavigator](https://news-navigator.labs.loc.gov/) (Historical Newspapers)
- [Math Formula Detection](http://transcriptorium.eu/~htrcontest/MathsICDAR2021/)

See [here](https://github.com/Layout-Parser/layout-parser/blob/master/docs/notes/modelzoo.md) for an overview. You will have to adapt the label map to conform to [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML) region (sub)types accordingly.

### [DocBank](https://github.com/doc-analysis/DocBank/blob/master/MODEL_ZOO.md)

X101-FPN [archive](https://layoutlm.blob.core.windows.net/docbank/model_zoo/X101.zip)

Proposed mappings:
- `["TextRegion:heading", "TextRegion:credit", "TextRegion:caption", "TextRegion:other", "MathsRegion", "GraphicRegion", "TextRegion:footer", "TextRegion:floating", "TextRegion:paragraph", "TextRegion:endnote", "TextRegion:heading", "TableRegion", "TextRegion:heading"]` (using only predefined `@type`)
- `["TextRegion:abstract", "TextRegion:author", "TextRegion:caption", "TextRegion:date", "MathsRegion", "GraphicRegion", "TextRegion:footer", "TextRegion:list", "TextRegion:paragraph", "TextRegion:reference", "TextRegion:heading", "TableRegion", "TextRegion:title"]` (using `@custom` as well)

## Testing

none yet


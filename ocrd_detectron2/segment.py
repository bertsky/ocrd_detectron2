from __future__ import absolute_import

import sys
import os
import tempfile
import fileinput
import shutil
import math
import multiprocessing as mp
import multiprocessing.sharedctypes
import ctypes
from typing import Optional

import numpy as np
# torch must be imported before shapely 1.8, see issue shapely#1598.
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cv2
from PIL import Image
#from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.utils import visualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog #, DatasetCatalog

from ocrd_utils import (
    resource_filename,
    getLogger,
    pushd_popd,
    coordinates_of_segment,
    coordinates_for_segment,
    crop_image,
    points_from_polygon,
    polygon_from_points,
)
from ocrd_models.ocrd_page import (
    OcrdPage,
    PageType,
    AdvertRegionType,
    ChartRegionType,
    ChemRegionType,
    CustomRegionType,
    GraphicRegionType,
    ImageRegionType,
    LineDrawingRegionType,
    MapRegionType,
    MathsRegionType,
    MusicRegionType,
    NoiseRegionType,
    SeparatorRegionType,
    TableRegionType,
    TextRegionType,
    UnknownRegionType,
    CoordsType,
    AlternativeImageType
)
from ocrd_models.ocrd_page_generateds import (
    ChartTypeSimpleType,
    GraphicsTypeSimpleType,
    TextTypeSimpleType
)
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage

# when doing Numpy postprocessing, enlarge masks via
# outer (convex) instead of inner (concave) hull of
# corresponding connected components
NP_POSTPROCESSING_OUTER = False
# when pruning overlapping detections (in either mode),
# require at least this share of the area to be redundant
RECALL_THRESHOLD = 0.8
# when finalizing contours of detections (in either mode),
# snap to connected components overlapping by this share
# (of component area), i.e. include if larger and exclude
# if smaller than this much
IOCC_THRESHOLD = 0.4
# when finalizing contours of detections (in either mode),
# add this many pixels in each direction
FINAL_DILATION = 4

class Detectron2Segment(Processor):
    max_workers = 1 # GPU context sharable across not forks

    @property
    def executable(self):
        return 'ocrd-detectron2-segment'

    def setup(self):
        #setup_logger(name='fvcore')
        #mp.set_start_method("spawn", force=True)
        # runtime overrides
        if self.parameter['device'] == 'cpu' or not torch.cuda.is_available():
            device = "cpu"
        else:
            device = self.parameter['device']
        self.logger.info("Using compute device %s", device)
        model_config = self.resolve_resource(self.parameter['model_config'])
        self.logger.info("Loading config '%s'", model_config)
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        with tempfile.TemporaryDirectory() as tmpdir:
            # workaround for fvcore/detectron2's stupid decision
            # to resolve the relative path for _BASE_ in the config file
            # on its dirname instead of the detectron2 distribution's config directory
            temp_configs = os.path.join(tmpdir, 'configs')
            with resource_filename('detectron2', 'model_zoo/configs') as stock_configs:
                shutil.copytree(stock_configs, temp_configs)
            temp_config = os.path.join(temp_configs, os.path.basename(model_config))
            shutil.copyfile(model_config, temp_config)
            with pushd_popd(tmpdir):
                # repair broken config files that make deviating assumptions on model_zoo files
                with fileinput.input(temp_config, inplace=True) as temp_config_file:
                    for line in temp_config_file:
                        if fileinput.isfirstline():
                            PREFIXES = ['/content/',
                                        '../../configs/',
                                        '../configs/',
                                        '../']
                            line = next((line.replace(pref, '') for pref in PREFIXES
                                         if line.startswith('_BASE_: "' + pref)), line)
                        if os.path.basename(model_config) == 'Jambo-sudo_X101.yaml' and 'NUM_CLASSES: 5' in line:
                            # workaround for Jambo-sudo/Historical-document-layout-analysis#1
                            line = line.replace('NUM_CLASSES: 5', 'NUM_CLASSES: 6')
                        print(line, end='')
                cfg = get_cfg()
                cfg.merge_from_file(temp_config)
        model_weights = self.resolve_resource(self.parameter['model_weights'])
        cfg.merge_from_list([
            # set threshold for this model
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.parameter['min_confidence'],
            "MODEL.RETINANET.SCORE_THRESH_TEST", self.parameter['min_confidence'],
            "MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH", self.parameter['min_confidence'],
            # or cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH ?
            "MODEL.DEVICE", device,
            "MODEL.WEIGHTS", model_weights,
        ])
        cfg.freeze()
        assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == len(self.parameter['categories']), \
            "The chosen model's number of classes %d does not match the given list of categories %d " % (
                cfg.MODEL.ROI_HEADS.NUM_CLASSES, len(self.parameter['categories']))
        # instantiate model
        self.logger.info("Loading weights '%s'", model_weights)
        self.predictor = DefaultPredictor(cfg)
        self.categories = self.parameter['categories']
        self.metadatacat = MetadataCatalog.get('runtime')
        self.metadatacat.thing_classes = self.categories

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Use detectron2 to segment each page into regions.

        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested
        ``operation_level``.

        Fetch a raw and a binarized image for the page/segment (possibly
        cropped and deskewed).

        Feed the raw image into the detectron2 predictor that has been
        used to load the given model. Then, depending on the model capabilities
        (whether it can do panoptic segmentation or only instance segmentation,
        whether the latter can do masks or only bounding boxes), post-process
        the predictions:

        \b
        - panoptic segmentation: take the provided segment label map, and
          apply the segment to class label map,
        - instance segmentation: find an optimal non-overlapping set (flat
          map) of instances via non-maximum suppression,
        - both: avoid overlapping pre-existing top-level regions (incremental
          segmentation).

        Then extend / shrink the surviving masks to fully include / exclude
        connected components in the foreground that are on the boundary.

        (This describes the steps when ``postprocessing`` is `full`. A value
        of `only-nms` will omit the morphological extension/shrinking, while
        `only-morph` will omit the non-maximum suppression, and `none` will
        skip all postprocessing.)

        Finally, find the convex hull polygon for each region, and map its
        class id to a new PAGE region type (and subtype).

        (Does not annotate `ReadingOrder` or `TextLine`s or `@orientation`.)

        Produce a new output file by serialising the resulting hierarchy.
        """
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        level = self.parameter['operation_level']

        page = pcgts.get_Page()
        page_image_raw, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_filter='binarized')
        # for morphological post-processing, we will need the binarized image, too
        if self.parameter['postprocessing'] != 'none':
            page_image_bin, _, _ = self.workspace.image_from_page(
                page, page_id, feature_selector='binarized')
            page_image_raw, page_image_bin = _ensure_consistent_crops(
                page_image_raw, page_image_bin)
        else:
            page_image_bin = page_image_raw
        # determine current zoom and target zoom
        if page_image_info.resolution != 1:
            dpi = page_image_info.resolution
            if page_image_info.resolutionUnit == 'cm':
                dpi = round(dpi * 2.54)
            zoom = 300.0 / dpi
        else:
            dpi = None
            zoom = 1.0
        # todo: if zoom is > 4.0, do something along the lines of eynollah's enhance
        if zoom < 2.0:
            # actual resampling: see below
            zoomed = zoom / 2.0
            self.logger.info("scaling %dx%d image by %.2f", page_image_raw.width, page_image_raw.height, zoomed)
        else:
            zoomed = 1.0

        for segment in ([page] if level == 'page' else
                        page.get_AllRegions(depth=1, classes=['Table'])):
            # regions = segment.get_AllRegions(depth=1)
            # FIXME: as long as we don't have get_AllRegions on region level,
            #        we have to simulate this via parent_object filtering
            def at_segment(region):
                return region.parent_object_ is segment
            regions = list(filter(at_segment, page.get_AllRegions()))

            if isinstance(segment, PageType):
                image_raw = page_image_raw
                image_bin = page_image_bin
                coords = page_coords
            else:
                image_raw, coords = self.workspace.image_from_segment(
                    segment, page_image_raw, page_coords, feature_filter='binarized')
                if self.parameter['postprocessing'] != 'none':
                    image_bin, _ = self.workspace.image_from_segment(
                        segment, page_image_bin, page_coords)
                    image_raw, image_bin = _ensure_consistent_crops(
                        image_raw, image_bin)
                else:
                    image_bin = image_raw

            # ensure RGB (if raw was merely grayscale)
            if image_raw.mode == '1':
                image_raw = image_raw.convert('L')
            image_raw = image_raw.convert(mode='RGB')
            image_bin = image_bin.convert(mode='1')

            # reduce resolution to 300 DPI max
            if zoomed != 1.0:
                image_bin = image_bin.resize(
                    (int(image_raw.width * zoomed),
                     int(image_raw.height * zoomed)),
                    resample=Image.Resampling.BICUBIC)
                image_raw = image_raw.resize(
                    (int(image_raw.width * zoomed),
                     int(image_raw.height * zoomed)),
                    resample=Image.Resampling.BICUBIC)

            # convert raw to BGR
            array_raw = np.array(image_raw)
            array_raw = array_raw[:,:,::-1]
            # convert binarized to single-channel negative
            array_bin = np.array(image_bin)
            array_bin = ~ array_bin

            image = self._process_segment(segment, regions, coords, array_raw, array_bin, zoomed, page_id)
            if image:
                result.images.append(image)
        return result

    def _process_segment(self, segment, ignore, coords, array_raw, array_bin, zoomed, page_id) -> Optional[OcrdPageResultImage]:
        self.logger = getLogger('processor.Detectron2Segment')
        cpu = torch.device('cpu')
        segtype = segment.__class__.__name__[:-4]
        # remove existing segmentation (have only detected targets survive)
        #page.set_ReadingOrder(None)
        #page.set_TextRegion([])
        segment.set_custom('coords=%s' % coords['transform'])
        height, width, _ = array_raw.shape
        postprocessing = self.parameter['postprocessing']
        scale = 43
        if postprocessing in ['full', 'only-morph']:
            # get connected components to estimate scale
            _, components = cv2.connectedComponents(array_bin.astype(np.uint8))
            # estimate glyph scale (roughly)
            _, counts = np.unique(components, return_counts=True)
            if counts.shape[0] > 1:
                counts = np.sqrt(3 * counts)
                counts = counts[(5 < counts) & (counts < 100)]
                scale = int(np.median(counts))
                self.logger.debug("estimated scale: %d", scale)
        # predict
        output = self.predictor(array_raw)
        if self.parameter['debug_img'] != 'none':
            vis = visualizer.Visualizer(array_raw,
                                        metadata=self.metadatacat,
                                        instance_mode={
                                            'instance_colors': visualizer.ColorMode.IMAGE,
                                            'instance_colors_only': visualizer.ColorMode.IMAGE_BW,
                                            'category_colors': visualizer.ColorMode.SEGMENTATION
                                        }[self.parameter['debug_img']])
        # decoding, cf. https://detectron2.readthedocs.io/en/latest/tutorials/models.html
        if 'panoptic_seg' in output:
            self.logger.info("decoding from panoptic segmentation results")
            segmap, seginfo = output['panoptic_seg']
            if not isinstance(segmap, np.ndarray):
                self.logger.debug(str(segmap))
                segmap = segmap.to(cpu)
                segmap = segmap.numpy()
            if self.parameter['debug_img'] != 'none':
                visimg = vis.draw_panoptic_seg(segmap, seginfo)
            seglabels = np.unique(segmap)
            nseg = len(seglabels)
            if not nseg:
                self.logger.warning("Detected no regions on %s '%s'", segtype, segment.id)
                return None
            masks = []
            classes = []
            scores = []
            for label in seglabels:
                if label == -1:
                    continue
                if seginfo is None:
                    class_id = label // self.predictor.metadata.label_divisor
                else:
                    for info in seginfo:
                        if info['id'] == label:
                            class_id = info['category_id']
                            break
                if not self.categories[class_id]:
                    continue
                masks.append(segmap == label)
                scores.append(1.0) #scores[i]
                classes.append(class_id)
            if not len(masks):
                self.logger.warning("Detected no regions for selected categories on %s '%s'", segtype, segment.id)
                return None
        elif 'instances' in output:
            self.logger.info("decoding from instance segmentation results")
            instances = output['instances']
            if not isinstance(instances, dict):
                assert instances.image_size == (height, width)
                instances = instances.to(cpu)
                if self.parameter['debug_img'] != 'none':
                    visimg = vis.draw_instance_predictions(instances)
                instances = instances.get_fields()
            classes = instances['pred_classes']
            if not all(self.categories):
                # filter out inactive classes
                select = np.array([bool(cat) for cat in self.categories])
                select = select[classes]
                for key, val in instances.items():
                    instances[key] = val[select]
                classes = instances['pred_classes']
            scores = instances['scores']
            if not isinstance(scores, np.ndarray):
                scores = scores.to(cpu).numpy()
            if not scores.shape[0]:
                self.logger.warning("Detected no regions on %s '%s'", segtype, segment.id)
                return None
            if 'pred_masks' in instances: # or pred_masks_rle ?
                masks = np.asarray(instances['pred_masks'])
                def get_mask(x):
                    # convert from RLE/polygon/Numpy # or Tensor?
                    # zzz tensor result would have to use .detach().numpy() ...
                    x = visualizer.GenericMask(x, height, width)
                    return x.mask > 0
                masks = np.stack([get_mask(x) for x in masks])
            elif 'pred_boxes' in instances:
                self.logger.warning("model has no mask output, only bbox")
                boxes = instances['pred_boxes']
                if not isinstance(boxes, np.ndarray):
                    boxes = boxes.to(cpu).tensor.numpy()
                assert boxes.shape[1] == 4 # and not 5 (rotated boxes)
                assert boxes.shape[0], "prediction without instances"
                masks = np.zeros((len(boxes), height, width), bool)
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    masks[i,
                          math.floor(y1):math.ceil(y2),
                          math.floor(x1):math.ceil(x2)] = True
        else:
            self.logger.error("Found no suitable output format to decode from")
            return None
        assert len(scores) == len(classes) == len(masks)
        # apply non-maximum suppression between overlapping instances
        # (not strictly necessary in case of panoptic segmentation,
        #  but we can still have overlaps with preexisting regions)
        if len(ignore):
            scores = np.insert(scores, 0, 1.0, axis=0)
            classes = np.insert(classes, 0, -1, axis=0)
            masks = np.insert(masks, 0, 0, axis=0)
            mask0 = np.zeros(masks.shape[1:], np.uint8)
            for i, region in enumerate(ignore):
                polygon = coordinates_of_segment(region, _, coords)
                if zoomed != 1.0:
                    polygon = np.round(polygon * zoomed).astype(int)
                cv2.fillPoly(mask0, pts=[polygon], color=(255,))
            assert np.count_nonzero(mask0), "existing regions all outside of page frame"
            masks[0] |= mask0 > 0
        if postprocessing in ['full', 'only-nms']:
            scores, classes, masks = postprocess_nms(
                scores, classes, masks, array_bin, self.categories,
                min_confidence=self.parameter['min_confidence'], nproc=8, logger=self.logger)
        if postprocessing in ['full', 'only-morph']:
            scores, classes, masks = postprocess_morph(
                scores, classes, masks, components, nproc=8, logger=self.logger)
        if len(ignore):
            scores = scores[1:]
            classes = classes[1:]
            masks = masks[1:]
        # convert to polygons and regions
        region_no = 0
        for mask, class_id, score in zip(masks, classes, scores):
            category = self.categories[class_id]
            # dilate until we have a single outer contour
            invalid = True
            for _ in range(10):
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 1 and len(contours[0]) > 3:
                    invalid = False
                    break
                mask = cv2.dilate(mask.astype(np.uint8),
                                  np.ones((scale,scale), np.uint8)) > 0
            if invalid:
                self.logger.warning("Ignoring non-contiguous (%d) region for %s", len(contours), category)
                continue
            region_polygon = contours[0][:,0,:] # already in x,y order
            if zoomed != 1.0:
                region_polygon = region_polygon / zoomed
            # ensure consistent and valid polygon outline
            region_polygon = coordinates_for_segment(region_polygon, _, coords)
            region_polygon = polygon_for_parent(region_polygon, segment)
            if region_polygon is None:
                self.logger.warning("Ignoring extant region for %s", category)
                continue
            # annotate new region/line
            region_coords = CoordsType(points_from_polygon(region_polygon), conf=score)
            cat2class = dict([
                ('AdvertRegion', AdvertRegionType),
                ('ChartRegion', ChartRegionType),
                ('ChemRegion', ChemRegionType),
                ('CustomRegion', CustomRegionType),
                ('GraphicRegion', GraphicRegionType),
                ('ImageRegion', ImageRegionType),
                ('LineDrawingRegion', LineDrawingRegionType),
                ('MapRegion', MapRegionType),
                ('MathsRegion', MathsRegionType),
                ('MusicRegion', MusicRegionType),
                ('NoiseRegion', NoiseRegionType),
                ('SeparatorRegion', SeparatorRegionType),
                ('TableRegion', TableRegionType),
                ('TextRegion', TextRegionType),
                ('UnknownRegion', UnknownRegionType),
                ])
            cat = category.split(':')
            try:
                regiontype = cat2class[cat[0]]
            except KeyError:
                raise ValueError("Invalid region type %s (see https://github.com/PRImA-Research-Lab/PAGE-XML)", cat[0])
            region_no += 1
            region_id = 'region%04d_%s' % (region_no, cat[0])
            region = regiontype(id=region_id, Coords=region_coords)
            if len(cat) > 1:
                try:
                    {TextRegionType: TextTypeSimpleType,
                     GraphicRegionType: GraphicsTypeSimpleType,
                     ChartRegionType: ChartTypeSimpleType}[regiontype](cat[1])
                    region.set_type(cat[1])
                except (KeyError, ValueError):
                    region.set_custom(cat[1])
            getattr(segment, 'add_' + cat[0])(region)
            self.logger.info("Detected %s region%04d (p=%.2f) on %s '%s'",
                     category, region_no, score, segtype, segment.id)
        if self.parameter['debug_img'] != 'none':
            altimg = AlternativeImageType(comments='debug')
            segment.add_AlternativeImage(altimg)
            return OcrdPageResultImage(
                Image.fromarray(visimg.get_image()),
                ('' if isinstance(segment, PageType) else '_' + segment.id) + '.IMG-DEBUG',
                altimg)
        return None


def postprocess_nms(scores, classes, masks, page_array_bin, categories, min_confidence=0.5, nproc=8, logger=None):
    """Apply geometrical post-processing to raw detections: remove overlapping candidates via non-maximum suppression across classes.

    Implement via Numpy routines.
    """
    if logger is None:
        logger = getLogger('ocrd.processor.Detectron2Segment')
    # apply IoU-based NMS across classes
    assert masks.dtype == bool
    instances = np.arange(len(masks))
    instances_i, instances_j = np.meshgrid(instances, instances, indexing='ij')
    combinations = list(zip(*np.where(instances_i != instances_j)))
    shared_masks = mp.sharedctypes.RawArray(ctypes.c_bool, masks.size)
    shared_masks_np = tonumpyarray_with_shape(shared_masks, masks.shape)
    np.copyto(shared_masks_np, masks * page_array_bin)
    with mp.Pool(processes=nproc, # to be refined via param
                 initializer=overlapmasks_init,
                 initargs=(shared_masks, masks.shape)) as pool:
        # multiprocessing for different combinations of array slices (pure)
        overlapping_combinations = pool.starmap(overlapmasks, combinations)
    overlaps = np.zeros((len(masks), len(masks)), bool)
    for (i, j), overlapping in zip(combinations, overlapping_combinations):
        if overlapping:
            overlaps[i, j] = True
    # find best-scoring instance per class
    bad = np.zeros_like(instances, bool)
    for i in np.argsort(-scores):
        score = scores[i]
        mask = masks[i]
        assert mask.shape[:2] == page_array_bin.shape[:2]
        ys, xs = mask.nonzero()
        assert xs.any() and ys.any(), "instance has empty mask"
        bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
        class_id = classes[i]
        if class_id < 0:
            logger.debug("ignoring existing region at %s", str(bbox))
            continue
        category = categories[class_id]
        if scores[i] < min_confidence:
            logger.debug("Ignoring instance for %s with too low score %.2f", category, score)
            bad[i] = True
            continue
        count = np.count_nonzero(mask)
        if count < 10:
            logger.warning("Ignoring too small (%dpx) region for %s", count, category)
            bad[i] = True
            continue
        worse = score < scores
        if np.any(worse & overlaps[i]):
            logger.debug("Ignoring instance for %s with %.2f overlapping better neighbour",
                      category, score)
            bad[i] = True
        else:
            logger.debug("post-processing prediction for %s at %s area %d score %f",
                      category, str(bbox), count, score)
    # post-process detections morphologically and decode to region polygons
    # does not compile (no OpenCV support):
    keep = np.nonzero(~ bad)[0]
    if not keep.size:
        return [], [], []
    keep = sorted(keep, key=lambda i: scores[i], reverse=True)
    scores = scores[keep]
    classes = classes[keep]
    masks = masks[keep]
    return scores, classes, masks

def postprocess_morph(scores, classes, masks, components, nproc=8, logger=None):
    """Apply morphological post-processing to raw detections: extend masks to avoid chopping off fg connected components.

    Implement via Numpy routines.
    """
    if logger is None:
        logger = getLogger('ocrd.processor.Detectron2Segment')
    shared_masks = mp.sharedctypes.RawArray(ctypes.c_bool, masks.size)
    shared_components = mp.sharedctypes.RawArray(ctypes.c_int32, components.size)
    shared_masks_np = tonumpyarray_with_shape(shared_masks, masks.shape)
    shared_components_np = tonumpyarray_with_shape(shared_components, components.shape)
    np.copyto(shared_components_np, components, casting='equiv')
    np.copyto(shared_masks_np, masks)
    with mp.Pool(processes=nproc, # to be refined via param
                 initializer=morphmasks_init,
                 initargs=(shared_masks, masks.shape,
                           shared_components, components.shape)) as pool:
        # multiprocessing for different slices of array (in-place)
        pool.map(morphmasks, range(masks.shape[0]))
    masks = tonumpyarray_with_shape(shared_masks, masks.shape)
    return scores, classes, masks

def polygon_for_parent(polygon, parent):
    """Clip polygon to parent polygon range.

    (Should be moved to ocrd_utils.coordinates_for_segment.)
    """
    childp = Polygon(polygon)
    if isinstance(parent, PageType):
        if parent.get_Border():
            parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
        else:
            parentp = Polygon([[0,0], [0,parent.get_imageHeight()],
                               [parent.get_imageWidth(),parent.get_imageHeight()],
                               [parent.get_imageWidth(),0]])
    else:
        parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    if not childp.is_valid:
        return None
    if not parentp.is_valid:
        return None
    # check if clipping is necessary
    if childp.within(parentp):
        return childp.exterior.coords[:-1]
    # clip to parent
    interp = childp.intersection(parentp)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        return None
    if interp.type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        # FIXME: construct concave hull / alpha shape
        interp = interp.convex_hull
    if interp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        interp = Polygon(np.round(interp.exterior.coords))
        interp = make_valid(interp)
    return interp.exterior.coords[:-1] # keep open

def make_valid(polygon):
    for split in range(1, len(polygon.exterior.coords)-1):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(polygon.exterior.coords[-split:]+polygon.exterior.coords[:-split])
    for tolerance in range(1, int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance)
    return polygon

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr, dtype=np.dtype(mp_arr))

def tonumpyarray_with_shape(mp_arr, shape):
    return np.frombuffer(mp_arr, dtype=np.dtype(mp_arr)).reshape(shape)

def overlapmasks_init(masks_array, masks_shape):
    global shared_masks
    global shared_masks_shape
    shared_masks = masks_array
    shared_masks_shape = masks_shape

def overlapmasks(i, j):
    # is i redundant w.r.t. j (i.e. j already covers most of its area)
    masks = np.ctypeslib.as_array(shared_masks).reshape(shared_masks_shape)
    imask = masks[i]
    jmask = masks[j]
    intersection = np.count_nonzero(imask * jmask)
    if not intersection:
        return False
    base = np.count_nonzero(imask)
    if intersection / base > RECALL_THRESHOLD:
        return True
    return False

def morphmasks_init(masks_array, masks_shape, components_array, components_shape):
    global shared_masks
    global shared_masks_shape
    global shared_components
    global shared_components_shape
    shared_masks = masks_array
    shared_masks_shape = masks_shape
    shared_components = components_array
    shared_components_shape = components_shape

def morphmasks(instance):
    masks = np.ctypeslib.as_array(shared_masks).reshape(shared_masks_shape)
    components = np.ctypeslib.as_array(shared_components).reshape(shared_components_shape)
    mask = masks[instance]
    # find closure in connected components
    complabels = np.unique(mask * components)
    left, top, w, h = cv2.boundingRect(mask.astype(np.uint8))
    right = left + w
    bottom = top + h
    if NP_POSTPROCESSING_OUTER:
        # overwrite pixel mask from (padded) outer bbox
        for label in complabels:
            if not label:
                continue # bg/white
            leftc, topc, wc, hc = cv2.boundingRect((components == label).astype(np.uint8))
            rightc = leftc + wc
            bottomc = topc + hc
            if wc > 2 * w or hc > 2 * h:
                continue # huge (non-text?) component
            # intersection over component too small?
            if (min(right, rightc) - max(left, leftc)) * \
                (min(bottom, bottomc) - max(top, topc)) < IOCC_THRESHOLD * wc * hc:
                continue # too little overlap
            newleft = min(left, leftc)
            newtop = min(top, topc)
            newright = max(right, rightc)
            newbottom = max(bottom, bottomc)
            if (newright - newleft) > 2 * w or (newbottom - newtop) > 1.5 * h:
                continue #
            left = newleft
            top = newtop
            right = newright
            bottom = newbottom
            w = right - left
            h = bottom - top
        left = max(0, left - FINAL_DILATION)
        top = max(0, top - FINAL_DILATION)
        right = min(mask.shape[1], right + FINAL_DILATION)
        bottom = min(mask.shape[0], bottom + FINAL_DILATION)
        mask[top:bottom, left:right] = True

    else:
        # fill pixel mask from (padded) inner bboxes
        for label in complabels:
            if not label:
                continue # bg/white
            suppress = False
            leftc, topc, wc, hc = cv2.boundingRect((components == label).astype(np.uint8))
            rightc = leftc + wc
            bottomc = topc + hc
            if wc > 2 * w or hc > 2 * h:
                # huge (non-text?) component
                suppress = True
            if (min(right, rightc) - max(left, leftc)) * \
                (min(bottom, bottomc) - max(top, topc)) < IOCC_THRESHOLD * wc * hc:
                # intersection over component too small
                suppress = True
            newleft = min(left, leftc)
            newtop = min(top, topc)
            newright = max(right, rightc)
            newbottom = max(bottom, bottomc)
            if (newright - newleft) > 2 * w or (newbottom - newtop) > 1.5 * h:
                # huge (non-text?) component
                suppress = True
            elif (newright - newleft) < 1.1 * w and (newbottom - newtop) < 1.1 * h:
                suppress = False
            if suppress:
                leftc = min(mask.shape[1], leftc + FINAL_DILATION)
                topc = min(mask.shape[0], topc + FINAL_DILATION)
                rightc = max(0, rightc - FINAL_DILATION)
                bottomc = max(0, bottomc - FINAL_DILATION)
                mask[topc:bottomc, leftc:rightc] = False
            else:
                leftc = max(0, leftc - FINAL_DILATION)
                topc = max(0, topc - FINAL_DILATION)
                rightc = min(mask.shape[1], rightc + FINAL_DILATION)
                bottomc = min(mask.shape[0], bottomc + FINAL_DILATION)
                mask[topc:bottomc, leftc:rightc] = True
                left = newleft
                top = newtop
                right = newright
                bottom = newbottom
                w = right - left
                h = bottom - top

def _ensure_consistent_crops(image_raw, image_bin):
    # workaround for OCR-D/core#687:
    if 0 < abs(image_raw.width - image_bin.width) <= 2:
        diff = image_raw.width - image_bin.width
        if diff > 0:
            image_raw = crop_image(
                image_raw,
                (int(np.floor(diff / 2)), 0,
                 image_raw.width - int(np.ceil(diff / 2)),
                 image_raw.height))
        else:
            image_bin = crop_image(
                image_bin,
                (int(np.floor(-diff / 2)), 0,
                 image_bin.width - int(np.ceil(-diff / 2)),
                 image_bin.height))
    if 0 < abs(image_raw.height - image_bin.height) <= 2:
        diff = image_raw.height - image_bin.height
        if diff > 0:
            image_raw = crop_image(
                image_raw,
                (0, int(np.floor(diff / 2)),
                 image_raw.width,
                 image_raw.height - int(np.ceil(diff / 2))))
        else:
            image_bin = crop_image(
                image_bin,
                (0, int(np.floor(-diff / 2)),
                 image_bin.width,
                 image_bin.height - int(np.ceil(-diff / 2))))
    return image_raw, image_bin

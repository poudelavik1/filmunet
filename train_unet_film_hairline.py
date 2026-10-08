"""Train a GSD-conditioned (FiLM) hairline-crack U-Net for thin and faint crack segmentation.

Expected dataset layout::

    dataset/
      train/images, train/masks
      val/images,   val/masks
      test/images,  test/masks

This is the FiLM U-Net counterpart of ``train_swin_unet_hairline.py``. Data
pairing, the reference augmentation, loss, metrics, checkpoints and figures
are identical, so the two models can be compared directly; the network, its
optimiser settings and everything that concerns the ground sampling distance
(GSD, mm per pixel) are new. Training and evaluation only: there is no
full-orthomosaic inference, skeletonisation or DXF export here.

The network (``HairlineFiLMUNet``):

* Encoder: an ImageNet-pretrained torchvision ResNet (``--encoder``, default
  ResNet-34) giving features at 1/2, 1/4, 1/8, 1/16 and 1/32 scale.
* Decoder: the reference bilinear-upsampling scSE ``DecoderBlock`` at 1/16,
  1/8, 1/4 and 1/2 scale.
* Hairline head: ResNet skips start at 1/2 scale, which is too coarse for 1-px
  cracks, so the reference full-resolution stem, fed with fixed morphological
  black-hat and top-hat responses (``ThinLineResponse``), supplies the skip of
  a final decoder block at 1/1 scale. ``--no-line-prior`` disables the
  thin-line channels.
* The decoder outputs at 1/8 and 1/4 scale are deeply supervised, with the
  reference auxiliary weights. The output biases start at the crack prior.
* FiLM conditioning (Perez et al., "FiLM: Visual Reasoning with a General
  Conditioning Layer", 2018). The same crack is a different number of pixels
  wide at a different GSD, so the network is told the GSD of each tile:
  normalized log(GSD) -> Linear(1, 64) -> SiLU -> Linear(64, 64) -> SiLU, then
  one Linear(64, 2C) per decoder block (and per skip connection with
  ``--film-skips``) gives a per-channel scale and shift,
  F' = (1 + d_gamma) * F + beta.
  - Every final Linear starts at zero, so d_gamma = 0 and beta = 0: training
    begins as a plain U-Net and learns to use the GSD gradually.
  - The input is (log(gsd) - mean) / std, with mean and std taken over the
    training tiles whose GSD is known and stored in the checkpoint.
  - Unknown GSD: a learned "null GSD" embedding replaces the output of the
    first Linear. During training a known GSD is also replaced by it with
    probability ``--gsd-dropout`` (0.15), so tiles without a GSD can be trained
    on and the model still runs when no GSD is given.
  - ``--no-film`` removes the conditioning: the plain hairline U-Net baseline
    for the ablation, identical otherwise.
* ``forward(images, gsd)``: ``gsd`` is a float tensor (N,) in mm/px, NaN for
  unknown. Eval mode returns logits (N, 1, H, W); train mode with deep
  supervision returns (logits, [auxiliary logits]).

GSD data (``--gsd-csv``): a CSV with the columns ``pattern,gsd_mm_per_px``.
``pattern`` is a regular expression searched in the image file name (for
example ``^EH104_``); the first matching row gives the GSD and a tile without
a match is unknown (NaN). A table of sources, tile counts and GSD values per
split is printed at start-up. Without ``--gsd-csv`` every tile is unknown and
FiLM has nothing to learn from.

GSD follows every scale change. An augmentation that enlarges the image by a
factor s shows each crack s times wider in pixels, so the tile's GSD becomes
gsd / s. That holds for the reference rescale (0.75-1.25) and affine
(0.9-1.1), the random resized crop, zoom-out, perspective (mean local scale at
the tile centre) and mosaic (gsd x 2). Crops, flips, 90-degree turns,
photometric changes, elastic and lens-type warps, cutout, fading, synthetic
cracks and the downscale-upscale degradation leave it unchanged, and an
unknown GSD stays unknown. Copy-paste donors, CutMix and MixUp partners and
mosaic tiles come from the same source family and the same GSD (or all
unknown), and are brought to the scale of the tile they are mixed into.

Kept from the reference, unchanged:

* Dataset pairing, Kaggle dataset discovery and the default paths.
* Native-resolution random crops for training (no resizing to 256 px), full
  unresized images for validation and test, tiled evaluation of large images
  with blended logits, evaluation batches grouped by shape and capped in
  megapixels.
* The reference augmentation and its defaults: rescale, crack-focused crop,
  crack copy-paste, synthetic hairlines, crack fading, flips and turns, affine,
  elastic, the photometric chain, cutout with the ``valid`` mask, and the
  contrast "all" mode with 21 levels and ``contrast_coverage.json``.
* Loss = BCE + recall-leaning Tversky + clDice with warm-up and deep
  supervision.
* Model selection and threshold calibration on validation only with the 2-px
  tolerance F1; strict precision, recall, F1 and IoU; per-source metrics; EMA
  weights; flip TTA; warm-up + cosine schedule, AMP, gradient clipping, early
  stopping, resumable ``last.pt`` / ``best.pt``, ``--evaluate-only``.
* Every table and figure, in the same plot style.

Changed from the reference:

* The network and its optimiser defaults (the CNN settings of
  ``train_unet_hairline.py``): learning rate 3e-4, encoder factor 0.1, weight
  decay 1e-4; ``--optimizer adam`` for plain Adam. The FiLM generator trains at
  the decoder rate without weight decay, which would pull it back to "ignore
  the GSD".
* ``--loss focal``: binary focal loss (alpha 0.25, gamma 2) in place of
  BCE + Tversky, with the same clDice term and deep supervision.
* The dataset returns (image, mask, valid, domain_id, index, gsd); rescale and
  affine also return their scale factor; copy-paste donors are restricted to
  the tile's GSD and resampled to its scale. The photometric chain is cut
  into its four stages (tone, blur, noise, JPEG) with unchanged code, so the
  new operations can run between them.
* ``--max-rotation`` may go up to 180 degrees (full rotation, reflect-padded).
* Test metrics are also reported per GSD value (or GSD bin), and
  ``--gsd-sanity-check`` re-runs the test set with the true GSD, GSD x 0.5,
  GSD x 2 and no GSD to show whether the model uses it.

Extended augmentation library. Every operation below is a registered function
with its own probability ``--<name>-p`` and, where it has one, a magnitude
``--<name>-m`` in [0, 1] that scales its range (1 = the full range named in
its help text). They are OFF unless a preset or a flag enables them.

* Geometric operations move image, mask and valid together: masks are warped
  linearly and cut at 0.4 so 1-px cracks stay connected, valid is warped with
  nearest neighbour, and pixels warped in from outside the image get valid = 0
  instead of a background label.
* An occlusion that hides a crack sets valid = 0 there.
* Label safety (``--label-safety``, on by default): after an operation that
  can reduce crack contrast (blur, noise, fog, compression, exposure, shadows,
  stains), the crack contrast against the local background is measured as in
  the copy-paste donor test. Where a crack that was visible has dropped below
  4 grey levels, its pixels and a 2-px band around them get valid = 0, so the
  model is never trained on a label it cannot see. The measure is taken per
  32-px cell. Cracks that were already below the limit in the source image are
  the dataset's own faint labels and are left as they are.
* Fixed order: scale -> crop -> mixing (mosaic / CutMix / MixUp, copy-paste)
  -> synthetic cracks and distractors -> geometric -> orthomosaic artefacts ->
  photometric -> weather and lighting -> blur -> noise -> compression ->
  occlusion.

Presets (``--aug-preset``); an individual ``--<name>-p`` / ``--<name>-m`` flag
overrides the preset:

* ``reference`` (default): the reference augmentation, nothing added. Its
  samples are identical to the reference script's for the same random seed,
  with one exception that the GSD requires: a crack pasted into a tile that
  was rescaled is resampled by the same factor, so that both have one GSD.
* ``none``: a random native-resolution crop (with the reference crack focus,
  which is sampling rather than augmentation) and nothing else.
* ``light``, ``medium``, ``heavy``: the reference plus the operations below.
  Magnitude of every enabled operation: 0.5, 0.75 and 1.0. ``--max-rotation``:
  20, 45 and 180 degrees.
* ``all``: every operation at probability 0.05 and magnitude 0.75 with
  ``--max-rotation`` 180, including those marked ``*``. A stress test of the
  pipeline, not a training recipe.

Probability of each operation per preset (``*`` = off in light, medium and
heavy because it harms crack labels; the reason is in the code)::

    operation            stage        light  medium  heavy  all
    mosaic               mixing       0      0.05    0.10   0.05
    cutmix               mixing       0      0.10    0.15   0.05
    mixup*               mixing       0      0       0      0.05
    branch-crack         cracks       0.04   0.06    0.08   0.05
    varwidth-crack       cracks       0.03   0.05    0.06   0.05
    joint                distractors  0.05   0.08    0.10   0.05
    sawcut               distractors  0      0.04    0.06   0.05
    scratch              distractors  0      0.04    0.06   0.05
    painted-line         distractors  0      0.03    0.05   0.05
    cable-shadow         distractors  0      0.05    0.08   0.05
    stain                distractors  0.05   0.08    0.10   0.05
    efflorescence        distractors  0      0.05    0.08   0.05
    dirt                 distractors  0      0.05    0.08   0.05
    spall                distractors  0      0.04    0.06   0.05
    resized-crop         rescaling    0      0.15    0.25   0.05
    zoom-out             rescaling    0      0.08    0.12   0.05
    shear                warp         0.10   0.15    0.20   0.05
    translate            warp         0.10   0.15    0.20   0.05
    perspective          warp         0.10   0.15    0.20   0.05
    optical              warp         0      0.08    0.12   0.05
    grid-distortion      warp         0      0.05    0.10   0.05
    piecewise-affine*    warp         0      0       0      0.05
    seam                 ortho        0.05   0.08    0.12   0.05
    nodata-border        ortho        0      0.03    0.05   0.05
    exposure-step        ortho        0.05   0.08    0.12   0.05
    blur-patch           ortho        0      0.05    0.08   0.05
    ghosting*            ortho        0      0       0      0.05
    rgb-shift            photometric  0.10   0.12    0.15   0.05
    channel-shuffle      photometric  0      0       0.02   0.05
    color-jitter         photometric  0.15   0.20    0.25   0.05
    white-balance        photometric  0.15   0.20    0.25   0.05
    tone-curve           photometric  0.10   0.15    0.20   0.05
    autocontrast         photometric  0      0.05    0.08   0.05
    equalize             photometric  0      0       0.05   0.05
    sharpen              photometric  0.05   0.08    0.10   0.05
    emboss               photometric  0      0       0.02   0.05
    posterize            photometric  0      0       0.02   0.05
    solarize*            photometric  0      0       0      0.05
    overexposure         photometric  0      0.04    0.06   0.05
    underexposure        photometric  0      0.04    0.06   0.05
    shadow               weather      0.08   0.12    0.15   0.05
    sun-flare            weather      0      0       0.03   0.05
    vignette             weather      0.05   0.08    0.10   0.05
    spotlight            weather      0      0.05    0.08   0.05
    fog                  weather      0      0       0.03   0.05
    wet                  weather      0      0.05    0.08   0.05
    rain*                weather      0      0       0      0.05
    defocus              blur         0.03   0.05    0.08   0.05
    median-blur          blur         0      0.02    0.04   0.05
    zoom-blur*           blur         0      0       0      0.05
    glass-blur*          blur         0      0       0      0.05
    poisson-noise        noise        0.08   0.10    0.15   0.05
    iso-noise            noise        0.05   0.08    0.10   0.05
    speckle              noise        0      0.05    0.08   0.05
    salt-pepper          noise        0      0       0.03   0.05
    pixel-dropout        noise        0      0.03    0.05   0.05
    chromatic-aberration noise        0      0.05    0.08   0.05
    ringing              noise        0      0.03    0.05   0.05
    webp                 compression  0.08   0.10    0.15   0.05
    downscale            compression  0.08   0.10    0.15   0.05
    random-erasing       occlusion    0.05   0.08    0.10   0.05
    gridmask             occlusion    0      0.04    0.06   0.05
    hide-and-seek        occlusion    0      0.04    0.06   0.05

``--randaugment N M`` additionally applies N randomly chosen photometric,
weather, blur, noise or compression operations (never geometric or mixing, and
never a ``*`` operation) at magnitude M (0-10) to every training image.

Augmentation tooling: ``--preview-augmentations K`` writes, for K training
tiles, one panel per enabled augmentation (image | label overlay | valid map,
with the GSD before and after) to ``<output>/augmentation_preview`` before
training; ``--preview-only`` stops there. ``augmentation_counts.json`` records
how often each augmentation fired and how many crack pixels the label-safety
check excluded. The resolved augmentation configuration is stored in
``configuration.json`` and ``best.pt``.

Outputs, in ``unet_film_hairline_output`` by default: the reference's
``configuration.json``, ``training_history.csv``, ``last.pt``, ``best.pt``,
``val_threshold_sweep.csv``, ``test_metrics.json``, ``contrast_coverage.json``,
``test_predictions/`` and ``plots/``, plus ``augmentation_counts.json``,
``augmentation_preview/`` and ``gsd_sanity_check.json`` when requested.
``best.pt`` also stores the architecture, the FiLM flag, the GSD
normalisation, the GSD table and the augmentation configuration.

``--self-test`` checks the FiLM identity at initialisation, the alignment,
binary labels and GSD bookkeeping of every augmentation, and the unknown-GSD
path through dataset, augmentation and model, then exits.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# The target dataset: every source in it counts (EH104 tiles are no longer singled out).
DATASET_NAME = "split_80_10_10"
LOCAL_DATASET_PATH = Path(r"E:\STRUCTURAL ENGINEERING\thesis\crackdetection\DATASETS") / DATASET_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
# torchvision ResNets: channels of the features at 1/2, 1/4, 1/8, 1/16 and 1/32 scale.
ENCODER_CHANNELS = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
}
METRIC_NAMES = ("precision", "recall", "f1", "iou", "tol_precision", "tol_recall", "tol_f1", "accuracy")
# Pixel counts accumulated per threshold; "near_*" are the 2-px tolerant hits.
COUNT_NAMES = ("tp", "fp", "fn", "near_label", "predicted", "near_prediction", "labelled", "pixels")
BASIC_METRICS = ("precision", "recall", "f1", "iou", "accuracy")  # logged for train and validation at 0.5


def find_dataset_root(base: Path = KAGGLE_INPUT_ROOT) -> Path:
    """Find the dataset among the Kaggle inputs.

    The folder named DATASET_NAME is preferred; otherwise the first input with
    the expected paired split layout is used.
    """
    if base.is_dir():
        candidates = [
            train_directory.parent for train_directory in sorted(base.rglob("train"))
            if (train_directory / "images").is_dir() and (train_directory / "masks").is_dir()
        ]
        named = [candidate for candidate in candidates if candidate.name == DATASET_NAME]
        if candidates:
            return (named or candidates)[0]
    return LOCAL_DATASET_PATH


DEFAULT_DATASET_PATH = find_dataset_root()
DEFAULT_OUTPUT_PATH = (
    KAGGLE_WORKING_ROOT / "unet_film_hairline_output"
    if KAGGLE_WORKING_ROOT.is_dir() else Path("unet_film_hairline_output")
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # NumPy's global RNG is not reseeded per DataLoader worker by PyTorch.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(0)


# --------------------------------------------------------------------------- #
# Dataset pairing (same rules as train_unet_hairline.py)
# --------------------------------------------------------------------------- #
def normalized_stem(path: Path) -> str:
    stem = path.stem.lower().strip()
    # Remove repeated/common role suffixes while preserving the tile identity.
    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r"(?:[_\- ](?:mask|label|image|img))$", "", stem)
    return stem


def tile_coordinate_key(path: Path) -> str | None:
    """Extract a stable row/column key when image and mask prefixes differ."""
    match = re.search(r"(?:^|[_\-])r(?:ow)?[_\-]?(\d+)[_\-]c(?:ol)?[_\-]?(\d+)", path.stem, re.IGNORECASE)
    if match:
        return f"r{int(match.group(1)):06d}_c{int(match.group(2)):06d}"
    return None


def unique_map(paths, key_function, description):
    result = {}
    duplicates = []
    for path in paths:
        key = key_function(path)
        if key is None:
            continue
        if key in result:
            duplicates.append((key, result[key].name, path.name))
        else:
            result[key] = path
    if duplicates:
        examples = "; ".join(f"{key}: {first}, {second}" for key, first, second in duplicates[:3])
        raise ValueError(f"Duplicate {description} tile identifiers: {examples}")
    return result


def collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    image_dir, mask_dir = split_dir / "images", split_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError(f"Missing images/masks folders in {split_dir}")
    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    mask_paths = sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    images = unique_map(image_paths, normalized_stem, "image")
    masks = unique_map(mask_paths, normalized_stem, "mask")
    if set(images) == set(masks):
        return [(images[key], masks[key]) for key in sorted(images)]

    # DXF tile datasets sometimes use different source prefixes for the image
    # and mask. Pair by the invariant r####_c#### coordinate in that case.
    coordinate_images = unique_map(image_paths, tile_coordinate_key, "image coordinate")
    coordinate_masks = unique_map(mask_paths, tile_coordinate_key, "mask coordinate")
    if coordinate_images and set(coordinate_images) == set(coordinate_masks):
        print(
            f"Pairing {len(coordinate_images)} tiles in {split_dir.name} by row/column "
            "because image and mask filename prefixes differ."
        )
        return [
            (coordinate_images[key], coordinate_masks[key])
            for key in sorted(coordinate_images)
        ]

    missing_mask_keys = sorted(set(images) - set(masks))
    missing_image_keys = sorted(set(masks) - set(images))
    image_examples = [images[key].name for key in missing_mask_keys[:5]]
    mask_examples = [masks[key].name for key in missing_image_keys[:5]]
    raise ValueError(
        f"Unpaired files in {split_dir}: {len(missing_mask_keys)} missing masks and "
        f"{len(missing_image_keys)} missing images. "
        f"Example unmatched images: {image_examples}. Example unmatched masks: {mask_examples}."
    )


def source_family(path: Path) -> str:
    """Group files by naming pattern, e.g. EH104_r0003_c0012 -> EH#_r#_c#."""
    return re.sub(r"\d+", "#", normalized_stem(path)) or "#"


def image_shape(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return height, width
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape[:2]


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return an RGB uint8 image and a {0, 1} uint8 mask of identical size."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or mask is None:
        raise ValueError(f"Could not read pair: {image_path}, {mask_path}")
    if image.shape[:2] != mask.shape:
        # Phone photos may carry an EXIF rotation that the mask did not get.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image.shape[:2] != mask.shape:
        print(f"Warning: resizing {image_path.name} {image.shape[:2]} to its mask {mask.shape}.")
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    # JPEG masks (e.g. CRACK500) contain compression noise; 0/1 masks also occur.
    threshold = 1 if mask.max() <= 1 else 128
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (mask >= threshold).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Ground sampling distance (GSD, mm per pixel)
# --------------------------------------------------------------------------- #
UNKNOWN_GSD = float("nan")
# Lower limit of the log(GSD) standard deviation used for normalisation. The
# scale augmentation alone spreads log(GSD) by about this much, so with one
# (or several almost equal) GSD values a smaller divisor would blow the FiLM
# input up to tens of standard deviations.
GSD_LOG_STD_FLOOR = 0.1


class GSDTable:
    """``pattern,gsd_mm_per_px`` rows: the first pattern found in a file name gives its GSD."""

    def __init__(self, rows=()):
        self.rows = [(str(pattern), float(gsd)) for pattern, gsd in rows]
        self.patterns = [re.compile(pattern) for pattern, _ in self.rows]

    @classmethod
    def from_csv(cls, path: Path) -> "GSDTable":
        rows = []
        # utf-8-sig: Excel writes a byte-order mark that would otherwise end up in the first column name.
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = {(name or "").strip().lower(): name for name in (reader.fieldnames or ())}
            if "pattern" not in columns or "gsd_mm_per_px" not in columns:
                raise SystemExit(f"{path} needs the columns pattern,gsd_mm_per_px (found {reader.fieldnames}).")
            for line, row in enumerate(reader, start=2):
                pattern = (row[columns["pattern"]] or "").strip()
                if not pattern or pattern.startswith("#"):
                    continue
                try:
                    gsd = float((row[columns["gsd_mm_per_px"]] or "").strip())
                    re.compile(pattern)
                except (ValueError, re.error) as exc:
                    raise SystemExit(f"{path}, line {line}: {exc}") from exc
                if not (math.isfinite(gsd) and gsd > 0):
                    raise SystemExit(f"{path}, line {line}: gsd_mm_per_px must be a positive number, not {gsd}.")
                rows.append((pattern, gsd))
        return cls(rows)

    def lookup(self, file_name: str) -> float:
        for pattern, (_, gsd) in zip(self.patterns, self.rows):
            if pattern.search(file_name):
                return gsd
        return UNKNOWN_GSD


def gsd_key(gsd: float):
    """Grouping key of a GSD value: None for unknown. Tiles are only mixed when their keys are equal."""
    return round(gsd, 6) if math.isfinite(gsd) else None


def format_gsd(gsd: float) -> str:
    return f"{gsd:.3g}" if math.isfinite(gsd) else "unknown"


def describe_gsd(values) -> str:
    """The distinct GSD values of a group of tiles, for the start-up table."""
    if not values:
        return "-"
    known = sorted({gsd_key(value) for value in values} - {None})
    parts = [f"{value:g}" for value in known[:3]]
    if len(known) > 3:
        parts.append(f"+{len(known) - 3} more")
    if any(not math.isfinite(value) for value in values):
        parts.append("unknown")
    return ", ".join(parts)


def gsd_normalization(values) -> dict:
    """Mean and standard deviation of log(GSD) over the tiles whose GSD is known.

    The logarithm makes the input symmetric in scale: half and double the GSD
    are equally far from the mean.
    """
    logs = np.log([value for value in values if math.isfinite(value)])
    if not len(logs):  # no GSD at all: the model only ever uses the null embedding
        return {"log_mean": 0.0, "log_std": 1.0, "known_tiles": 0, "distinct_values": 0}
    return {
        "log_mean": float(logs.mean()), "log_std": float(max(logs.std(), GSD_LOG_STD_FLOOR)),
        "known_tiles": int(len(logs)), "distinct_values": len({gsd_key(value) for value in values} - {None}),
    }


def gsd_groups(values, max_values: int = 12, bins: int = 6) -> tuple[list[str], list[int]]:
    """Groups for the per-GSD metrics: one per GSD value, or log-spaced bins when there are many.

    Returns the group labels (the last one is "unknown") and each tile's group index.
    """
    known = sorted({gsd_key(value) for value in values} - {None})
    if len(known) <= max_values:
        position = {value: index for index, value in enumerate(known)}
        labels = [f"{value:g} mm/px" for value in known]
        ids = [position.get(gsd_key(value), len(known)) for value in values]
    else:
        edges = np.geomspace(known[0], known[-1], bins + 1)
        labels = [f"{low:.3g}-{high:.3g} mm/px" for low, high in zip(edges[:-1], edges[1:])]
        ids = [int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, bins - 1))
               if math.isfinite(value) else bins for value in values]
    return labels + ["unknown"], ids


def print_source_table(domains, families, gsds, target_domains) -> None:
    """Source family, number of tiles and GSD value(s) per split."""
    width = max(12, *(len(describe_gsd(gsds[name])) for name in families))
    print(f"{'Source family':30s} " + " ".join(f"{name:>6s} {'GSD mm/px':<{width}s}" for name in families))
    for domain in domains:
        cells = []
        for name in families:
            values = [gsd for family, gsd in zip(families[name], gsds[name]) if family == domain]
            cells.append(f"{len(values):6d} {describe_gsd(values):<{width}s}")
        print(f"{domain:30s} " + " ".join(cells) + ("  <- target" if domain in target_domains else ""))


# --------------------------------------------------------------------------- #
# Augmentation. `valid` marks supervised pixels; padding, warped-in borders and
# cutout regions are excluded from the loss instead of being labelled background.
# This first part is the reference's, unchanged except that rescale() and
# random_affine() also return their scale factor, which the GSD follows.
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationOptions:
    scale_p: float = 0.30
    focus_p: float = 0.60
    copy_paste_p: float = 0.20
    synthetic_p: float = 0.08
    fade_p: float = 0.25
    affine_p: float = 0.30
    elastic_p: float = 0.10
    cutout_p: float = 0.10
    max_rotation: float = 20.0
    synthetic_label_width: int = 4
    # Contrast enhancement / reduction of 1-10 %.
    #   "all":    every training image is cycled through every level, from the
    #             largest reduction to the largest enhancement in contrast_step
    #             steps: -10 %, -9 %, ..., 0 (original), ..., +10 % = 21 levels.
    #   "random": with probability contrast_p, a random change in that range,
    #             raised or lowered 50/50.
    #   "off":    no contrast change (not in the reference; the "none" preset).
    contrast_p: float = 0.60
    contrast_min_change: float = 0.01
    contrast_max_change: float = 0.10
    contrast_mode: str = "all"
    contrast_step: float = 0.01
    # ----- Not in the reference. With these defaults the augmentation is the reference's. -----
    dihedral: bool = True      # flips and 90-degree turns; off only in the "none" preset
    photometric: bool = True   # the reference photometric chain; off only in the "none" preset
    # Extended library: {operation name: (probability, magnitude)}; a missing operation is off.
    ops: dict = field(default_factory=dict)
    label_safety: bool = True
    randaugment: tuple | None = None   # (N operations per image, magnitude M in 0-10)


def rescale(image, mask, valid, factor: float):
    """Also returns the scale factor applied: the caller divides the tile's GSD by it."""
    height, width = mask.shape
    size = (max(32, round(width * factor)), max(32, round(height * factor)))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, size, interpolation=interpolation)
    # Resample labels as coverage, then use a low cut so thin lines stay connected.
    mask = (cv2.resize(mask.astype(np.float32), size, interpolation=interpolation) >= 0.4).astype(np.uint8)
    valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    # The size is rounded to whole pixels, so report the scale that was really applied
    # (the geometric mean of the two axes) rather than the requested factor.
    return image, mask, valid, math.sqrt((size[0] / width) * (size[1] / height))


def pad_to_at_least(image, mask, valid, size: int, random_placement: bool = True):
    height, width = mask.shape
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if not pad_h and not pad_w:
        return image, mask, valid
    top = random.randint(0, pad_h) if random_placement else 0
    left = random.randint(0, pad_w) if random_placement else 0
    border = (top, pad_h - top, left, pad_w - left)
    image = cv2.copyMakeBorder(image, *border, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, *border, cv2.BORDER_CONSTANT, value=0)
    valid = cv2.copyMakeBorder(valid, *border, cv2.BORDER_CONSTANT, value=0)
    return image, mask, valid


def random_crop(image, mask, valid, size: int, focus_p: float):
    image, mask, valid = pad_to_at_least(image, mask, valid, size)
    height, width = mask.shape
    if random.random() < focus_p and mask.any():
        # Centre near a random crack pixel so large images still yield crack crops.
        ys, xs = np.nonzero(mask)
        pick = random.randrange(len(ys))
        centre_y = ys[pick] + random.randint(-size // 4, size // 4)
        centre_x = xs[pick] + random.randint(-size // 4, size // 4)
        y0 = int(np.clip(centre_y - size // 2, 0, height - size))
        x0 = int(np.clip(centre_x - size // 2, 0, width - size))
    else:
        y0, x0 = random.randint(0, height - size), random.randint(0, width - size)
    window = (slice(y0, y0 + size), slice(x0, x0 + size))
    return image[window], mask[window], valid[window]


def dihedral(image, mask, valid):
    if random.random() < 0.5:
        image, mask, valid = image[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    if random.random() < 0.5:
        image, mask, valid = image[::-1], mask[::-1], valid[::-1]
    turns = random.randint(0, 3)
    if turns:
        image, mask, valid = np.rot90(image, turns), np.rot90(mask, turns), np.rot90(valid, turns)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask), np.ascontiguousarray(valid)


def random_affine(image, mask, valid, max_rotation: float):
    """Also returns its scale factor (0.9-1.1): the caller divides the tile's GSD by it.

    Rotation and translation do not change the pixel size. `max_rotation` may be 180 for a
    full rotation: the image border is reflected and the warped-in pixels are not supervised.
    """
    height, width = mask.shape
    angle, scale = random.uniform(-max_rotation, max_rotation), random.uniform(0.9, 1.1)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += (random.uniform(-0.05, 0.05) * width, random.uniform(-0.05, 0.05) * height)
    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    # Linear warp + low cut avoids the staircase gaps nearest-neighbour leaves in 1-px lines.
    mask = cv2.warpAffine(mask.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(valid, matrix, (width, height), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, (mask >= 0.4).astype(np.uint8), valid, scale


def elastic(image, mask, valid):
    """Low-magnitude elastic warp that bends cracks without breaking them."""
    height, width = mask.shape
    alpha, sigma = min(height, width) * 0.012, min(height, width) * 0.035
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (height, width)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (height, width)).astype(np.float32), (0, 0), sigma) * alpha
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x, map_y = grid_x + dx, grid_y + dy
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(valid, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, (mask >= 0.4).astype(np.uint8), valid


def local_background(image: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Median filtering removes line structures narrower than about kernel/2."""
    return cv2.medianBlur(image, kernel)


def fade_cracks(image, mask, probability: float):
    """Blend labelled crack pixels toward the local background to mimic faint cracks."""
    if random.random() >= probability or not mask.any():
        return image
    background = local_background(image)
    region = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32)
    alpha = cv2.GaussianBlur(region, (5, 5), 0)[..., None] * random.uniform(0.30, 0.65)
    return (image * (1 - alpha) + background * alpha).astype(np.uint8)


def add_synthetic_hairline(image, mask, probability: float, label_width: int):
    """Draw a faint meandering 1-2 px line, dark or light, labelled like the DXF masks."""
    if random.random() >= probability:
        return image, mask
    height, width = mask.shape
    x, y = random.uniform(0, width), random.uniform(0, height)
    angle = random.uniform(0, 2 * math.pi)
    points = [(x, y)]
    for _ in range(random.randint(8, 25)):
        angle += random.gauss(0, 0.35)
        step = random.uniform(6, 14)
        x, y = x + step * math.cos(angle), y + step * math.sin(angle)
        if not (0 <= x < width and 0 <= y < height):
            break
        points.append((x, y))
    if len(points) < 3:
        return image, mask
    polyline = [np.round(np.asarray(points)).astype(np.int32)]
    line = np.zeros(mask.shape, np.float32)
    cv2.polylines(line, polyline, False, 1.0, random.choice((1, 1, 2)), cv2.LINE_AA)
    line = (line * random.uniform(0.25, 0.55))[..., None]
    image = image.astype(np.float32)
    # EH104 cracks are often lighter than the concrete, so both polarities occur.
    image = image * (1 - line) if random.random() < 0.5 else image + (255 - image) * line
    image = image.astype(np.uint8)
    mask = mask.copy()
    cv2.polylines(mask, polyline, False, 1, label_width, cv2.LINE_8)
    return image, mask


def cutout(image, valid):
    height, width = valid.shape
    erase_h, erase_w = random.randint(8, max(9, height // 8)), random.randint(8, max(9, width // 8))
    y, x = random.randint(0, height - erase_h), random.randint(0, width - erase_w)
    image = image.copy()
    valid = valid.copy()
    image[y:y + erase_h, x:x + erase_w] = image.mean(axis=(0, 1)).astype(np.uint8)
    valid[y:y + erase_h, x:x + erase_w] = 0
    return image, valid


def motion_blur(image: np.ndarray) -> np.ndarray:
    size = random.choice((3, 5))
    kernel = np.zeros((size, size), np.float32)
    if random.random() < 0.5:
        kernel[size // 2, :] = 1 / size
    else:
        kernel[:, size // 2] = 1 / size
    return cv2.filter2D(image, -1, kernel)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale the deviation of every pixel from the per-channel mean: factor 1.10 is
    a 10 % contrast enhancement, 0.90 a 10 % reduction. Brightness is unchanged."""
    mean = image.mean(axis=(0, 1), keepdims=True)
    return (image - mean) * factor + mean


def random_contrast_factor(minimum_change: float, maximum_change: float) -> float:
    """1 +/- a random change in [minimum_change, maximum_change], sign chosen 50/50."""
    change = random.uniform(minimum_change, maximum_change)
    return 1.0 + change if random.random() < 0.5 else 1.0 - change


def contrast_level_factors(options: AugmentationOptions) -> list[float]:
    """Every contrast level of the "all" mode as a factor, lowest first.

    With the defaults: 0.90, 0.91, ..., 0.99, 1.00 (original), 1.01, ..., 1.10.
    """
    steps = int(round((options.contrast_max_change - options.contrast_min_change) / options.contrast_step)) + 1
    changes = [round(options.contrast_min_change + step * options.contrast_step, 6) for step in range(steps)]
    return [round(1 - change, 6) for change in reversed(changes)] + [1.0] + [round(1 + change, 6) for change in changes]


# The reference photometric chain, in the four stages of the pipeline order (photometric, blur,
# noise, compression). The code and its random draws are the reference's; it is only cut into
# functions so that the new operations of each stage can run between them.
def reference_tone(image: np.ndarray, options: AugmentationOptions,
                   contrast_factor: float | None = None) -> np.ndarray:
    """Contrast level, brightness, gamma, illumination gradient, CLAHE, HSV and grayscale."""
    image = image.astype(np.float32)
    # Enhancement sharpens cracks; reduction compresses the crack/background
    # difference, which trains the model on fainter cracks.
    if contrast_factor is not None:
        if contrast_factor != 1.0:
            image = adjust_contrast(image, contrast_factor)
    elif random.random() < options.contrast_p:
        image = adjust_contrast(
            image, random_contrast_factor(options.contrast_min_change, options.contrast_max_change)
        )
    if random.random() < 0.5:
        image = image + random.uniform(-25, 25)  # brightness shift
    if random.random() < 0.3:
        image = 255.0 * np.clip(image / 255.0, 0, 1) ** random.uniform(0.7, 1.5)
    if random.random() < 0.2:
        # Smooth illumination gradient (shadow / uneven lighting on the surface).
        height, width = image.shape[:2]
        theta = random.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        ramp = (xx * math.cos(theta) + yy * math.sin(theta))
        ramp = (ramp - ramp.min()) / max(1e-6, float(ramp.max() - ramp.min()))
        image *= (random.uniform(0.6, 0.9) + (1 - random.uniform(0.6, 0.9)) * ramp)[..., None]
    image = np.clip(image, 0, 255).astype(np.uint8)
    if random.random() < 0.15:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(random.uniform(1.5, 3.0), (8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if random.random() < 0.3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-6, 6)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.10:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return image


def reference_blur(image: np.ndarray) -> np.ndarray:
    # Blur is kept mild: a 5x5 Gaussian can erase a 1-px crack while its label remains.
    blur_choice = random.random()
    if blur_choice < 0.08:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.4, 0.9))
    elif blur_choice < 0.12:
        image = motion_blur(image)
    return image


def reference_noise(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.25:
        noise = np.random.normal(0, random.uniform(2.0, 8.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def reference_jpeg(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.25:
        quality = random.randint(40, 90)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


def photometric_augment(image: np.ndarray, options: AugmentationOptions | None = None,
                        contrast_factor: float | None = None) -> np.ndarray:
    """`contrast_factor` is the level chosen by the "all" mode; None means "random" mode.

    The whole reference chain in one call; the training pipeline calls the four stages itself.
    """
    options = options or AugmentationOptions()
    image = reference_tone(image, options, contrast_factor)
    return reference_jpeg(reference_noise(reference_blur(image)))


# --------------------------------------------------------------------------- #
# Extended augmentation library. Every operation is registered with a stage,
# takes (tile, magnitude) and edits the tile in place. All of them are off
# unless a preset or a --<name>-p flag enables them.
# --------------------------------------------------------------------------- #
class Tile:
    """A training tile on its way through the augmentation pipeline.

    `scale` is the accumulated scale factor relative to the source image: a
    tile enlarged by s shows each crack s times wider in pixels, so its GSD is
    gsd / s (an unknown GSD, NaN, stays NaN). `partner()` returns the image,
    mask and valid of another tile of the same source family and GSD at the
    same scale, for the mixing operations. `before` is the image as it was
    before the first contrast-reducing operation, for the label-safety check.
    """

    __slots__ = ("image", "mask", "valid", "scale", "partner", "label_width", "before")

    def __init__(self, image, mask, valid, scale: float = 1.0, partner=None, label_width: int = 4):
        self.image, self.mask, self.valid = image, mask, valid
        self.scale, self.partner, self.label_width, self.before = scale, partner, label_width, None

    def copy(self) -> "Tile":
        return Tile(self.image.copy(), self.mask.copy(), self.valid.copy(), self.scale, self.partner,
                    self.label_width)


@dataclass(frozen=True)
class Augmentation:
    name: str
    stage: str
    function: object
    help: str
    scaled: bool         # has a magnitude (--<name>-m)
    label_safety: bool   # can hide a crack, so the label-safety check follows it
    default_off: bool    # harms crack labels: only its own flag or the "all" preset enables it


# Stages in pipeline order. The reference operations sit between them (see CrackSegmentationDataset.augment).
STAGES = ("mixing", "cracks", "distractors", "rescaling", "warp", "ortho",
          "photometric", "weather", "blur", "noise", "compression", "occlusion")
# RandAugment draws from these stages only: it never moves or mixes pixels.
RANDAUGMENT_STAGES = ("photometric", "weather", "blur", "noise", "compression")
AUGMENTATIONS: dict[str, Augmentation] = {}


def register(name: str, stage: str, help: str, scaled: bool = True, label_safety: bool = False,
             default_off: bool = False):
    def decorator(function):
        assert stage in STAGES and name not in AUGMENTATIONS
        AUGMENTATIONS[name] = Augmentation(name, stage, function, help, scaled, label_safety, default_off)
        return function
    return decorator


def amount(m: float, low: float, high: float) -> float:
    """A random strength between `low` and `low + m * (high - low)`: the magnitude scales the range."""
    return random.uniform(low, low + m * (high - low))


def signed(m: float, limit: float) -> float:
    return random.uniform(-limit, limit) * m


def to_uint8(image) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def resize_tile(image, mask, valid, width: int, height: int):
    """rescale() to an exact size, with its rules: labels resampled as coverage and cut at 0.4."""
    shrinking = width * height < mask.shape[0] * mask.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    mask = (cv2.resize(mask.astype(np.float32), (width, height), interpolation=interpolation) >= 0.4).astype(np.uint8)
    valid = cv2.resize(valid, (width, height), interpolation=cv2.INTER_NEAREST)
    return image, mask, valid


def warp_matrix(tile: Tile, matrix: np.ndarray) -> None:
    """Affine (2x3) or perspective (3x3) warp of image, mask and valid, as in random_affine().

    Image: linear with a reflected border. Mask: linear, then the 0.4 cut that
    keeps 1-px cracks connected. Valid: nearest neighbour, and 0 for every
    pixel warped in from outside the image, which therefore leaves the loss
    instead of being labelled background.
    """
    height, width = tile.mask.shape
    warp = cv2.warpAffine if matrix.shape[0] == 2 else cv2.warpPerspective
    tile.image = warp(tile.image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)
    mask = warp(tile.mask.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tile.mask = (mask >= 0.4).astype(np.uint8)
    tile.valid = warp(tile.valid, matrix, (width, height), flags=cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_maps(tile: Tile, map_x: np.ndarray, map_y: np.ndarray) -> None:
    """Free-form warp (source coordinates per output pixel), with the rules of warp_matrix()."""
    map_x, map_y = np.ascontiguousarray(map_x, np.float32), np.ascontiguousarray(map_y, np.float32)
    tile.image = cv2.remap(tile.image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(tile.mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tile.mask = (mask >= 0.4).astype(np.uint8)
    tile.valid = cv2.remap(tile.valid, map_x, map_y, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def pixel_grid(height: int, width: int):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return yy, xx


def smooth_noise(height: int, width: int, cell: float, octaves: int = 3) -> np.ndarray:
    """Perlin-type noise in [0, 1]: random grids of doubling resolution, cubic-upsampled and summed."""
    noise, weight = np.zeros((height, width), np.float32), 1.0
    for octave in range(octaves):
        step = max(2.0, cell / 2 ** octave)
        grid = np.random.rand(int(height / step) + 2, int(width / step) + 2).astype(np.float32)
        noise += weight * cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        weight *= 0.5
    low, high = float(noise.min()), float(noise.max())
    return (noise - low) / max(1e-6, high - low)


def soft_blobs(height: int, width: int, count: int, low: float = 0.08, high: float = 0.35) -> np.ndarray:
    """Float map in [0, 1] of `count` random filled ellipses with blurred edges."""
    blobs = np.zeros((height, width), np.float32)
    for _ in range(count):
        axes = (int(random.uniform(low, high) * width / 2) + 2, int(random.uniform(low, high) * height / 2) + 2)
        centre = (random.randint(0, width - 1), random.randint(0, height - 1))
        cv2.ellipse(blobs, centre, axes, random.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(blobs, (0, 0), max(1.0, 0.03 * min(height, width)))


def line_distance(height: int, width: int, axis_aligned: float = 0.5) -> np.ndarray:
    """Signed distance of every pixel to a random straight line through the tile's interior."""
    if random.random() < axis_aligned:  # photo footprints are mostly aligned with the mosaic grid
        angle = random.choice((0.0, math.pi / 2)) + math.radians(random.uniform(-3, 3))
    else:
        angle = random.uniform(0, math.pi)
    x, y = random.uniform(0.2, 0.8) * width, random.uniform(0.2, 0.8) * height
    yy, xx = pixel_grid(height, width)
    return (xx - x) * math.sin(angle) - (yy - y) * math.cos(angle)


def straight_line(height: int, width: int, angle: float):
    """End points, beyond the tile, of a straight line through a random interior point."""
    x, y = random.uniform(0.1, 0.9) * width, random.uniform(0.1, 0.9) * height
    return line_through(x, y, angle, height + width)


def line_through(x: float, y: float, angle: float, reach: float, offset: float = 0.0):
    """End points of the line through (x, y), `reach` px to each side; `offset` moves it sideways."""
    x, y = x - offset * math.sin(angle), y + offset * math.cos(angle)
    dx, dy = math.cos(angle) * reach, math.sin(angle) * reach
    return (int(round(x - dx)), int(round(y - dy))), (int(round(x + dx)), int(round(y + dy)))


def line_alpha(height: int, width: int, lines, thickness: int, blur: float = 0.0) -> np.ndarray:
    alpha = np.zeros((height, width), np.float32)
    for start, end in lines:
        cv2.line(alpha, start, end, 1.0, thickness, cv2.LINE_AA)
    return cv2.GaussianBlur(alpha, (0, 0), blur) if blur > 0 else alpha


def blend(image: np.ndarray, alpha: np.ndarray, colour) -> np.ndarray:
    """Blend `image` toward `colour` by the per-pixel `alpha`."""
    alpha = alpha[..., None]
    return to_uint8(image * (1 - alpha) + np.asarray(colour, np.float32) * alpha)


def exclude_hidden_cracks(tile: Tile, cover: np.ndarray) -> None:
    """Crack pixels under an opaque overlay cannot be seen: take them, and a 2-px band, out of the loss."""
    hidden = ((tile.mask > 0) & cover).astype(np.uint8)
    if hidden.any():
        tile.valid = tile.valid * (cv2.dilate(hidden, np.ones((5, 5), np.uint8)) == 0)


# ----- Label safety -----
# Visibility limit in grey levels: the one paste_donor_crack uses to reject donor cracks.
LABEL_SAFETY_GREY_LEVELS = 4.0
LABEL_SAFETY_CELL = 32


def label_safety_check(before: np.ndarray, after: np.ndarray, mask: np.ndarray, valid: np.ndarray,
                       threshold: float = LABEL_SAFETY_GREY_LEVELS, cell: int = LABEL_SAFETY_CELL):
    """Take cracks that an augmentation has made invisible out of the loss.

    Crack contrast is measured as in paste_donor_crack: the image minus its
    15-px median background, averaged over the crack pixels. It is taken per
    `cell` x `cell` block instead of over the whole tile, because a local blur
    patch or a shadow hides only part of a crack. A block is lost when its
    cracks were visible in `before` (|mean difference| >= threshold) and are
    not in `after`; its crack pixels and a 2-px band around them get valid = 0.

    Blocks that were below the threshold already in `before` are not touched.
    Those are the dataset's own faint labels (a 1-px hairline under a 4-px
    label line averages to a few grey levels), which the reference trains on
    and which are the cracks this model is for.

    Returns (valid, crack pixels excluded, crack pixels checked).
    """
    ys, xs = np.nonzero((mask > 0) & (valid > 0))
    if not len(ys):
        return valid, 0, 0
    cells = (ys // cell) * (mask.shape[1] // cell + 1) + xs // cell
    limit = threshold * np.bincount(cells)

    def visible(image):
        difference = image.astype(np.float32) - local_background(image, 15).astype(np.float32)
        return np.abs(np.bincount(cells, weights=difference[ys, xs].mean(1))) >= limit

    lost = (visible(before) & ~visible(after))[cells]
    if not lost.any():
        return valid, 0, len(ys)
    hidden = np.zeros(mask.shape, np.uint8)
    hidden[ys[lost], xs[lost]] = 1
    valid = valid * (cv2.dilate(hidden, np.ones((5, 5), np.uint8)) == 0)  # 5x5 = the pixels and a 2-px band
    return valid, int(lost.sum()), len(ys)


# ----- Mixing (one of these per tile at most; copy-paste is the reference's) -----
@register("mosaic", "mixing", "Four tiles of one source and GSD in a 2x2 grid, each at half size (GSD x 2)",
          scaled=False)
def mosaic(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    half_h, half_w = height // 2, width // 2
    parts = [(tile.image, tile.mask, tile.valid)] + [tile.partner() for _ in range(3)]
    random.shuffle(parts)
    image, mask, valid = np.zeros_like(tile.image), np.zeros_like(tile.mask), np.zeros_like(tile.valid)
    for position, part in enumerate(parts):
        window = (slice((position // 2) * half_h, (position // 2 + 1) * half_h),
                  slice((position % 2) * half_w, (position % 2 + 1) * half_w))
        image[window], mask[window], valid[window] = resize_tile(*part, half_w, half_h)
    tile.image, tile.mask, tile.valid = image, mask, valid
    # Every tile is shown at half size, so one pixel now covers twice the ground: GSD x 2.
    tile.scale *= 0.5


@register("cutmix", "mixing", "Paste a rectangle (20-60 %% of each side) of image, mask and valid from another tile")
def cutmix(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    box_h, box_w = int(round(amount(m, 0.2, 0.6) * height)), int(round(amount(m, 0.2, 0.6) * width))
    y, x = random.randint(0, height - box_h), random.randint(0, width - box_w)
    window = (slice(y, y + box_h), slice(x, x + box_w))
    partner_image, partner_mask, partner_valid = tile.partner()
    image, mask, valid = tile.image.copy(), tile.mask.copy(), tile.valid.copy()
    # The mask and valid rectangles are pasted with the image, so labels stay hard 0/1.
    image[window], mask[window], valid[window] = partner_image[window], partner_mask[window], partner_valid[window]
    tile.image, tile.mask, tile.valid = image, mask, valid


# OFF by default. MixUp blends two images and their labels. For a binary crack
# mask the blended label is soft and ambiguous: the partner's crack shows at a
# fraction of its contrast, and neither "crack" nor "background" is right for
# it. The losses and metrics here need hard labels, so the pixels where the two
# masks disagree are excluded (valid = 0), which removes exactly the crack
# pixels the tile was meant to teach.
@register("mixup", "mixing", "Blend with another tile at a weight up to 0.5; disagreeing labels are excluded",
          default_off=True)
def mixup(tile: Tile, m: float) -> None:
    partner_image, partner_mask, partner_valid = tile.partner()
    share = amount(m, 0.1, 0.5)
    tile.image = to_uint8(tile.image * (1 - share) + partner_image.astype(np.float32) * share)
    disagree = cv2.dilate((tile.mask != partner_mask).astype(np.uint8), np.ones((5, 5), np.uint8))
    tile.valid = tile.valid * partner_valid * (disagree == 0)


# ----- Synthetic cracks and hard negatives -----
def meander(x: float, y: float, angle: float, steps: int, height: int, width: int, wobble: float = 0.35):
    """A meandering path like the one add_synthetic_hairline draws, as a list of points."""
    points = [(x, y)]
    for _ in range(steps):
        angle += random.gauss(0, wobble)
        step = random.uniform(6, 14)
        x, y = x + step * math.cos(angle), y + step * math.sin(angle)
        if not (0 <= x < width and 0 <= y < height):
            break
        points.append((x, y))
    return points


def paint_crack(tile: Tile, strokes) -> None:
    """Draw strokes [(points, width in px)] as one faint crack, dark or light, and label them.

    Contrast and polarity follow add_synthetic_hairline; the label is as wide as the reference's
    synthetic label, or as the stroke where that is wider.
    """
    strokes = [(np.round(np.asarray(points)).astype(np.int32), width) for points, width in strokes]
    line = np.zeros(tile.mask.shape, np.float32)
    for points, width in strokes:
        cv2.polylines(line, [points], False, 1.0, width, cv2.LINE_AA)
    line = (line * random.uniform(0.25, 0.55))[..., None]
    image = tile.image.astype(np.float32)
    image = image * (1 - line) if random.random() < 0.5 else image + (255 - image) * line
    tile.image = image.astype(np.uint8)
    mask = tile.mask.copy()
    for points, width in strokes:
        cv2.polylines(mask, [points], False, 1, max(tile.label_width, width), cv2.LINE_8)
    tile.mask = mask


@register("branch-crack", "cracks", "Synthetic crack with 1-2 short side branches", scaled=False)
def branch_crack(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    trunk = meander(random.uniform(0, width), random.uniform(0, height), random.uniform(0, 2 * math.pi),
                    random.randint(8, 25), height, width)
    if len(trunk) < 4:
        return
    thickness = random.choice((1, 1, 2))
    strokes = [(trunk, thickness)]
    for _ in range(random.randint(1, 2)):
        node = random.randrange(1, len(trunk) - 1)
        heading = math.atan2(trunk[node + 1][1] - trunk[node][1], trunk[node + 1][0] - trunk[node][0])
        heading += random.choice((-1, 1)) * math.radians(random.uniform(30, 70))
        branch = meander(*trunk[node], heading, random.randint(2, 6), height, width, wobble=0.25)
        if len(branch) >= 2:
            strokes.append((branch, 1))  # a branch is never wider than its trunk
    paint_crack(tile, strokes)


@register("varwidth-crack", "cracks", "Synthetic crack whose width varies between 1 and 4 px along its length",
          scaled=False)
def varwidth_crack(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    path = meander(random.uniform(0, width), random.uniform(0, height), random.uniform(0, 2 * math.pi),
                   random.randint(8, 25), height, width)
    if len(path) < 4:
        return
    # Width changes smoothly between three random control widths, like a crack that opens and closes.
    profile = np.interp(np.linspace(0, 1, len(path) - 1), (0, 0.5, 1), [random.uniform(1, 4) for _ in range(3)])
    paint_crack(tile, [(path[index:index + 2], int(round(profile[index]))) for index in range(len(path) - 1)])


# Hard negatives: drawn into the image and labelled BACKGROUND (the mask is not
# touched). They teach the model that "line-shaped" is not the same as "crack".
# Every one of them is ruler-straight, of constant width and, except for the
# scratches, runs from edge to edge, so none can pass for a real crack, which
# meanders and varies in width. A distractor that looked like a crack would
# teach the model to miss cracks.
@register("joint", "distractors", "Straight construction joint or formwork line, 2-4 px, edge to edge")
def construction_joint(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    angle = random.choice((0.0, math.pi / 2)) + math.radians(random.uniform(-4, 4))
    x, y = random.uniform(0.1, 0.9) * width, random.uniform(0.1, 0.9) * height
    lines = [line_through(x, y, angle, height + width)]
    if random.random() < 0.4:  # the next formwork board
        lines.append(line_through(x, y, angle, height + width, random.uniform(40, 120)))
    alpha = line_alpha(height, width, lines, random.randint(2, 4)) * amount(m, 0.15, 0.5)
    tile.image = blend(tile.image, alpha, (0, 0, 0) if random.random() < 0.75 else (255, 255, 255))


@register("sawcut", "distractors", "Straight saw-cut groove, 4-8 px wide, dark with light rims")
def saw_cut(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    line = [straight_line(height, width, random.uniform(0, math.pi))]
    groove = random.randint(4, 8)
    rim = line_alpha(height, width, line, groove + 2) * amount(m, 0.1, 0.3)
    core = line_alpha(height, width, line, groove) * amount(m, 0.3, 0.7)
    tile.image = blend(blend(tile.image, rim, (255, 255, 255)), core, (0, 0, 0))


@register("scratch", "distractors", "Bundle of 2-5 parallel, straight, 1-px light scratches")
def scratches(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    angle = random.uniform(0, math.pi)
    x, y = random.uniform(0.2, 0.8) * width, random.uniform(0.2, 0.8) * height
    reach, spacing = random.uniform(0.2, 0.35) * min(height, width), random.uniform(3, 8)
    lines = [line_through(x, y, angle, reach, index * spacing) for index in range(random.randint(2, 5))]
    alpha = line_alpha(height, width, lines, 1) * amount(m, 0.15, 0.45)
    tile.image = blend(tile.image, alpha, (255, 255, 255))


@register("painted-line", "distractors", "Opaque straight painted line, 6-16 px wide")
def painted_line(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    line = [straight_line(height, width, random.uniform(0, math.pi))]
    colour = random.choice(((235, 235, 230), (230, 200, 60), (190, 50, 45), (50, 80, 170)))
    # Worn paint: the opacity varies with a noise texture.
    alpha = line_alpha(height, width, line, random.randint(6, 16)) * (0.75 + 0.25 * smooth_noise(height, width, 12))
    alpha *= amount(m, 0.6, 1.0)
    tile.image = blend(tile.image, alpha, colour)
    exclude_hidden_cracks(tile, alpha >= 0.5)  # paint covers the crack: its label is not visible any more


@register("cable-shadow", "distractors", "Soft straight shadow of a cable or rebar, 4-12 px wide",
          label_safety=True)
def cable_shadow(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    line = [straight_line(height, width, random.uniform(0, math.pi))]
    alpha = line_alpha(height, width, line, random.randint(4, 12), blur=random.uniform(1.5, 3.0))
    tile.image = blend(tile.image, alpha * amount(m, 0.15, 0.45), (0, 0, 0))


# Surface distractors: background as well, but blob-shaped, so they do not need to be straight.
@register("stain", "distractors", "Water stains / dark blotches: blurred ellipses darkened by up to 45 %%",
          label_safety=True)
def stain(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    alpha = soft_blobs(height, width, random.randint(1, 3)) * (0.6 + 0.4 * smooth_noise(height, width, 24))
    tile.image = blend(tile.image, alpha * amount(m, 0.15, 0.45), (0, 0, 0))


@register("efflorescence", "distractors", "Efflorescence-like light patches", label_safety=True)
def efflorescence(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    alpha = soft_blobs(height, width, random.randint(1, 3)) * (0.5 + 0.5 * smooth_noise(height, width, 16))
    tile.image = blend(tile.image, alpha * amount(m, 0.2, 0.55), (245, 245, 240))


@register("dirt", "distractors", "Dirt / moss texture overlay from Perlin-type noise", label_safety=True)
def dirt(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    cover = smooth_noise(height, width, random.uniform(24, 64), octaves=4)
    alpha = np.clip((cover - random.uniform(0.45, 0.65)) / 0.2, 0, 1) * (0.6 + 0.4 * smooth_noise(height, width, 6))
    colour = random.choice(((70, 85, 45), (95, 75, 50), (60, 60, 55)))  # moss, soil, soot
    tile.image = blend(tile.image, alpha * amount(m, 0.15, 0.5), colour)


@register("spall", "distractors", "Small spalls / pits: 3-12 dark dots", label_safety=True)
def spall(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    alpha = np.zeros((height, width), np.float32)
    for _ in range(random.randint(3, 12)):
        centre = (random.randint(0, width - 1), random.randint(0, height - 1))
        axes = (random.randint(1, 5), random.randint(1, 5))
        cv2.ellipse(alpha, centre, axes, random.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8) * amount(m, 0.3, 0.7)
    tile.image = blend(tile.image, alpha, (0, 0, 0))


# ----- Geometric -----
# Full 360-degree rotation needs no operation of its own: random_affine already
# reflect-pads, so --max-rotation 180 gives it.
@register("resized-crop", "rescaling",
          "Random resized crop: 60-100 %% of the area, aspect 0.8-1.25, resized back (changes GSD)")
def resized_crop(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    area = 1.0 - amount(m, 0.0, 0.4)
    aspect = math.exp(random.uniform(math.log(0.8), math.log(1.25)) * m)
    crop_w = int(np.clip(round(width * math.sqrt(area * aspect)), 16, width))
    crop_h = int(np.clip(round(height * math.sqrt(area / aspect)), 16, height))
    y, x = random.randint(0, height - crop_h), random.randint(0, width - crop_w)
    window = (slice(y, y + crop_h), slice(x, x + crop_w))
    tile.image, tile.mask, tile.valid = resize_tile(tile.image[window], tile.mask[window], tile.valid[window],
                                                    width, height)
    # The two axes are enlarged by different factors. Their geometric mean is the factor by which
    # areas, and a crack of average direction, grow, so that is the scale the GSD is divided by.
    tile.scale *= math.sqrt((width / crop_w) * (height / crop_h))


@register("zoom-out", "rescaling",
          "Shrink to 50-90 %% on a reflect-padded canvas; the padding leaves the loss (changes GSD)")
def zoom_out(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    image, mask, valid, factor = rescale(tile.image, tile.mask, tile.valid, 1.0 - amount(m, 0.1, 0.5))
    pad_h, pad_w = height - mask.shape[0], width - mask.shape[1]
    top, left = random.randint(0, pad_h), random.randint(0, pad_w)
    border = (top, pad_h - top, left, pad_w - left)
    # As in pad_to_at_least: the image is reflected so the network sees texture, not a black frame,
    # but the reflection holds mirrored cracks without labels, so it is not supervised.
    tile.image = cv2.copyMakeBorder(image, *border, cv2.BORDER_REFLECT_101)
    tile.mask = cv2.copyMakeBorder(mask, *border, cv2.BORDER_CONSTANT, value=0)
    tile.valid = cv2.copyMakeBorder(valid, *border, cv2.BORDER_CONSTANT, value=0)
    tile.scale *= factor


@register("shear", "warp", "Shear of up to +/-10 degrees along x or y")
def shear(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    slope = math.tan(math.radians(signed(m, 10.0)))
    if random.random() < 0.5:
        matrix = np.float32([[1, slope, -slope * height / 2], [0, 1, 0]])
    else:
        matrix = np.float32([[1, 0, 0], [slope, 1, -slope * width / 2]])
    # A shear has determinant 1: areas, and with them the mean scale and the GSD, do not change.
    warp_matrix(tile, matrix)


@register("translate", "warp", "Independent x and y shift of up to 10 %% of the tile")
def translate(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    warp_matrix(tile, np.float32([[1, 0, signed(m, 0.10) * width], [0, 1, signed(m, 0.10) * height]]))


def homography_scale(matrix: np.ndarray, x: float, y: float) -> float:
    """Local scale of a homography at the source point (x, y): the square root of its area magnification."""
    h = np.asarray(matrix, np.float64)
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    u, v = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
    jacobian = np.array([[h[0, 0] - u * h[2, 0], h[0, 1] - u * h[2, 1]],
                         [h[1, 0] - v * h[2, 0], h[1, 1] - v * h[2, 1]]]) / w
    return math.sqrt(abs(np.linalg.det(jacobian)))


@register("perspective", "warp", "Mild off-nadir perspective: each corner shifted by up to 4 %% (changes GSD)")
def perspective(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    shift = np.float32([[signed(m, 0.04) * width, signed(m, 0.04) * height] for _ in range(4)])
    matrix = cv2.getPerspectiveTransform(corners, corners + shift)
    warp_matrix(tile, matrix)
    # A perspective view has a different scale at every pixel, but a tile has one GSD. The scale
    # at the tile centre is its mean to first order, so the centre value updates the GSD.
    source = np.linalg.solve(matrix.astype(np.float64), np.array([width / 2, height / 2, 1.0]))
    tile.scale *= homography_scale(matrix, source[0] / source[2], source[1] / source[2])


@register("optical", "warp", "Lens barrel / pincushion distortion, radial coefficient up to +/-0.06")
def optical_distortion(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    coefficient = signed(m, 0.06)
    yy, xx = pixel_grid(height, width)
    half_w, half_h = (width - 1) / 2, (height - 1) / 2
    x, y = (xx - half_w) / half_w, (yy - half_h) / half_h
    factor = 1 + coefficient * (x * x + y * y)
    # The radial factor is 1 at the tile centre, so the scale there, and the tile's GSD, stay as
    # they are; the small coefficient keeps the scale error at the corners within a few percent.
    warp_maps(tile, x * factor * half_w + half_w, y * factor * half_h + half_h)


def distorted_axis(length: int, cells: int, limit: float) -> np.ndarray:
    """Source coordinate of every output pixel along one axis: `cells` cells of random relative width."""
    widths = np.array([1 + random.uniform(-limit, limit) for _ in range(cells)])
    knots = np.concatenate(([0.0], np.cumsum(widths))) * (length - 1) / widths.sum()
    return np.interp(np.arange(length), np.linspace(0, length - 1, cells + 1), knots).astype(np.float32)


@register("grid-distortion", "warp", "Grid distortion: 5x5 cells stretched or squeezed by up to 20 %%")
def grid_distortion(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    # The cell widths are renormalised to the full tile, so the frame stays in place and the
    # mean scale is 1: no pixel comes from outside and the GSD does not change.
    map_x = np.tile(distorted_axis(width, 5, 0.2 * m), (height, 1))
    map_y = np.tile(distorted_axis(height, 5, 0.2 * m)[:, None], (1, width))
    warp_maps(tile, map_x, map_y)


# OFF by default. Each triangle of the control grid gets its own affine map, so
# the warp has a kink along every triangle edge. A crack that crosses an edge
# is bent at a sharp corner there, a shape neither real cracks nor the label
# lines have; elastic() bends cracks smoothly instead.
@register("piecewise-affine", "warp", "Piecewise-affine warp on a 4x4 grid, control points moved by up to 4 %%",
          default_off=True)
def piecewise_affine(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    cells = 4
    shift = np.zeros((cells + 1, cells + 1, 2), np.float32)  # the border points stay, so the frame stays
    shift[1:-1, 1:-1] = np.random.uniform(-1, 1, (cells - 1, cells - 1, 2)) * 0.04 * m * min(height, width)
    yy, xx = pixel_grid(height, width)
    grid_x, grid_y = xx * cells / width, yy * cells / height
    column, row = grid_x.astype(np.int32), grid_y.astype(np.int32)
    u, v = (grid_x - column)[..., None], (grid_y - row)[..., None]
    p00, p10 = shift[row, column], shift[row, column + 1]
    p01, p11 = shift[row + 1, column], shift[row + 1, column + 1]
    # Each grid cell is split into two triangles along its diagonal.
    lower = p00 + u * (p10 - p00) + v * (p01 - p00)
    upper = p11 + (1 - u) * (p01 - p11) + (1 - v) * (p10 - p11)
    offset = np.where(u + v <= 1, lower, upper)
    warp_maps(tile, xx + offset[..., 0], yy + offset[..., 1])


# ----- Orthomosaic artefacts (UAV-specific) -----
# The seam and the no-data border change the mask, so they run first in this
# stage, before any operation that opens a label-safety comparison.
@register("seam", "ortho", "Stitching seam: one side of a random line shifted by 1-2 px")
def stitching_seam(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    distance = line_distance(height, width)
    side = distance > 0
    shift = random.randint(1, max(1, int(round(2 * m))))
    matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    matrix[random.randrange(2), 2] = random.choice((-shift, shift))
    size, nearest = (width, height), cv2.INTER_NEAREST
    image = cv2.warpAffine(tile.image, matrix, size, flags=nearest, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(tile.mask, matrix, size, flags=nearest, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(tile.valid, matrix, size, flags=nearest, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # A whole-pixel shift moves the label exactly with its crack, so a crack that crosses the seam
    # keeps a correct (shifted) label on both sides. On the seam itself the two photos overlap or
    # tear, and what the true crack does there is undefined: a 2-px band is not supervised.
    tile.image = np.where(side[..., None], image, tile.image)
    tile.mask = np.where(side, mask, tile.mask)
    tile.valid = np.where(side, valid, tile.valid) * (np.abs(distance) > 2)


@register("nodata-border", "ortho", "Black no-data region along one edge, 4-30 %% deep, as at an orthomosaic border")
def nodata_border(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    yy, xx = pixel_grid(height, width)
    edge = random.randrange(4)
    along, across, length, depth = (yy, xx, height, width) if edge < 2 else (xx, yy, width, height)
    if edge % 2:
        across = depth - 1 - across
    # Orthomosaic borders are rarely parallel to the tile grid, so the edge is slanted.
    region = across < amount(m, 0.04, 0.30) * depth + random.uniform(-0.2, 0.2) * (along - length / 2)
    image, mask, valid = tile.image.copy(), tile.mask.copy(), tile.valid.copy()
    # No image, so no visible crack: the label is background, and it is supervised (valid = 1)
    # because "predict nothing on no-data" is exactly what the model should learn here.
    image[region], mask[region], valid[region] = 0, 0, 1
    tile.image, tile.mask, tile.valid = image, mask, valid


@register("exposure-step", "ortho", "Brightness / contrast step across a seam between two differently exposed photos",
          label_safety=True)
def exposure_step(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    weight = np.clip(line_distance(height, width) / 1.5 + 0.5, 0, 1)[..., None]  # a seam about 1.5 px soft
    image = tile.image.astype(np.float32)
    exposed = image * (1 + signed(m, 0.2)) + signed(m, 25)
    tile.image = to_uint8(image * (1 - weight) + exposed * weight)


@register("blur-patch", "ortho", "Local blur patch: a mis-registered or motion-blurred source photo region",
          label_safety=True)
def blur_patch(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    weight = np.clip(soft_blobs(height, width, 1, 0.3, 0.7) * 1.5, 0, 1)[..., None]
    if random.random() < 0.5:
        blurred = cv2.GaussianBlur(tile.image, (5, 5), amount(m, 0.7, 1.6))
    else:
        blurred = motion_blur(tile.image)
    tile.image = to_uint8(tile.image * (1 - weight) + blurred.astype(np.float32) * weight)


# OFF by default. A ghost (double image from mis-registered photos) repeats
# every crack a few pixels away, and the copy has no label: the model would be
# told that a visible crack is background. The ghost of each labelled crack is
# excluded below, which makes the operation safe to try, but half-visible
# double cracks are still not a pattern worth teaching.
@register("ghosting", "ortho", "Ghosting / double image: a copy shifted by 2-6 px blended in at up to 40 %%",
          label_safety=True, default_off=True)
def ghosting(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    matrix = np.float32([[1, 0, random.choice((-1, 1)) * random.randint(2, 6)],
                         [0, 1, random.choice((-1, 1)) * random.randint(2, 6)]])
    ghost = cv2.warpAffine(tile.image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    ghost_mask = cv2.warpAffine(tile.mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    share = amount(m, 0.15, 0.4)
    tile.image = to_uint8(tile.image * (1 - share) + ghost.astype(np.float32) * share)
    unlabelled = (cv2.dilate(ghost_mask, np.ones((3, 3), np.uint8)) > 0) & (tile.mask == 0)
    tile.valid = tile.valid * ~unlabelled


# ----- Photometric and colour (after the reference's contrast, brightness, gamma, CLAHE, HSV) -----
LUMA = np.float32((0.299, 0.587, 0.114))


@register("rgb-shift", "photometric", "Independent offset of up to +/-20 grey levels per colour channel")
def rgb_shift(tile: Tile, m: float) -> None:
    tile.image = to_uint8(tile.image + np.float32([signed(m, 20) for _ in range(3)]))


@register("channel-shuffle", "photometric", "Random permutation of the colour channels", scaled=False)
def channel_shuffle(tile: Tile, m: float) -> None:
    tile.image = np.ascontiguousarray(tile.image[..., random.sample(range(3), 3)])


@register("color-jitter", "photometric",
          "Brightness and contrast +/-20 %%, saturation +/-30 %%, hue +/-9 degrees, in random order",
          label_safety=True)
def color_jitter(tile: Tile, m: float) -> None:
    def brightness(image):
        return image * (1 + signed(m, 0.2))

    def contrast(image):
        return adjust_contrast(image, 1 + signed(m, 0.2))

    def saturation(image):
        gray = (image @ LUMA)[..., None]
        return gray + (image - gray) * (1 + signed(m, 0.3))

    def hue(image):
        hsv = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(round(signed(m, 4.5)))) % 180  # OpenCV hue runs 0-179
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    image = tile.image.astype(np.float32)
    for operation in random.sample((brightness, contrast, saturation, hue), 4):
        image = operation(image)
    tile.image = to_uint8(image)


def blackbody_rgb(kelvin: float) -> np.ndarray:
    """sRGB colour of a black body (Tanner Helland's fit of the Planckian locus), 0-255 per channel."""
    t = kelvin / 100.0
    red = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    green = (99.4708025861 * math.log(t) - 161.1195681661 if t <= 66
             else 288.1221695283 * (t - 60) ** -0.0755148492)
    blue = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return np.clip(np.float32((red, green, blue)), 1.0, 255.0)


@register("white-balance", "photometric",
          "Planckian jitter: colour temperature between about 3600 K and 11800 K instead of 6500 K")
def white_balance(tile: Tile, m: float) -> None:
    # Illuminants (sun, shade, overcast sky) lie on the black-body locus, so shifting along it
    # gives the casts a camera's white balance really produces, unlike a free RGB shift.
    gains = blackbody_rgb(6500.0 * math.exp(signed(m, 0.6))) / blackbody_rgb(6500.0)
    tile.image = to_uint8(tile.image * (gains / float(gains @ LUMA)))  # brightness is left alone


@register("tone-curve", "photometric", "Random monotone tone curve: quarter points moved by up to +/-0.1",
          label_safety=True)
def tone_curve(tile: Tile, m: float) -> None:
    knots = np.linspace(0, 1, 5)
    values = knots + np.array([0, signed(m, 0.1), signed(m, 0.1), signed(m, 0.1), 0])
    values = np.maximum.accumulate(np.clip(values, 0, 1))  # a tone curve never reverses grey levels
    table = to_uint8(np.interp(np.arange(256) / 255.0, knots, values) * 255)
    tile.image = table[tile.image]


@register("autocontrast", "photometric",
          "Stretch the 1st-99th luminance percentiles toward the full range (gain capped at 2.5)", scaled=False)
def autocontrast(tile: Tile, m: float) -> None:
    low, high = np.percentile(tile.image @ LUMA, (1, 99))
    if high - low < 8:
        return
    # Concrete tiles are flat: a full stretch would be a gain of 5-10, which turns sensor noise
    # into texture. The gain is capped, and the brightness moves to mid-grey only as far as the
    # stretch was carried out.
    full, centre = 255.0 / (high - low), (low + high) / 2
    gain = min(full, 2.5)
    target = centre + (127.5 - centre) * (gain - 1) / max(full - 1, 1e-6)
    tile.image = to_uint8((tile.image.astype(np.float32) - centre) * gain + target)


@register("equalize", "photometric", "Histogram equalisation of the luminance, blended in at 15-60 %%",
          label_safety=True)
def equalize(tile: Tile, m: float) -> None:
    lab = cv2.cvtColor(np.ascontiguousarray(tile.image), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(np.ascontiguousarray(lab[:, :, 0]))
    # Full equalisation of a flat concrete tile is close to a binarisation, so it is only blended in.
    share = amount(m, 0.15, 0.6)
    tile.image = to_uint8(tile.image * (1 - share) + cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) * share)


@register("sharpen", "photometric", "Unsharp mask with a gain of up to 1.0")
def sharpen(tile: Tile, m: float) -> None:
    image = tile.image.astype(np.float32)
    tile.image = to_uint8(image + amount(m, 0.2, 1.0) * (image - cv2.GaussianBlur(image, (0, 0), 1.0)))


@register("emboss", "photometric", "Emboss: a directional relief added at up to 50 %%")
def emboss(tile: Tile, m: float) -> None:
    relief = cv2.filter2D(tile.image @ LUMA, -1, np.float32([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]))
    tile.image = to_uint8(tile.image + amount(m, 0.1, 0.5) * relief[..., None])


@register("posterize", "photometric", "Posterize to 4-7 bits per channel", label_safety=True)
def posterize(tile: Tile, m: float) -> None:
    drop = 8 - random.randint(max(4, 7 - int(round(3 * m))), 7)
    tile.image = (tile.image >> drop) << drop


# OFF by default. Solarising inverts every pixel above a threshold. A crack that
# crosses from a dark to a bright part of the surface then flips from darker to
# lighter than its surroundings in mid-line, and bright concrete becomes dark:
# an appearance no camera produces.
@register("solarize", "photometric", "Invert the pixels above a threshold of 128-255", default_off=True,
          label_safety=True)
def solarize(tile: Tile, m: float) -> None:
    threshold = 255 - amount(m, 0, 127)
    tile.image = np.where(tile.image >= threshold, 255 - tile.image, tile.image).astype(np.uint8)


@register("overexposure", "photometric", "Overexposure: gain of 1.2-1.9, highlights clip to white",
          label_safety=True)
def overexposure(tile: Tile, m: float) -> None:
    tile.image = to_uint8(tile.image * (1 + amount(m, 0.2, 0.9)))


@register("underexposure", "photometric", "Underexposure: black level raised by 15-70, shadows crush to black",
          label_safety=True)
def underexposure(tile: Tile, m: float) -> None:
    tile.image = to_uint8((tile.image - amount(m, 15, 70)) * random.uniform(0.7, 1.0))


# ----- Weather and lighting -----
@register("shadow", "weather", "Cast shadow: 1-2 soft-edged polygons darkened by 15-50 %%", label_safety=True)
def cast_shadow(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    shade = np.zeros((height, width), np.float32)
    for _ in range(random.randint(1, 2)):
        centre = np.array((random.uniform(0, width), random.uniform(0, height)))
        angles = np.sort(np.random.uniform(0, 2 * math.pi, random.randint(3, 6)))
        radii = np.random.uniform(0.25, 0.7, (len(angles), 1)) * max(height, width)
        points = centre + radii * np.stack((np.cos(angles), np.sin(angles)), 1)
        cv2.fillPoly(shade, [np.round(points).astype(np.int32)], 1.0, cv2.LINE_AA)
    shade = cv2.GaussianBlur(shade, (0, 0), random.uniform(1.0, 6.0)) * amount(m, 0.15, 0.5)
    tile.image = blend(tile.image, shade, (0, 0, 0))


def radial_distance(height: int, width: int, x: float, y: float) -> np.ndarray:
    yy, xx = pixel_grid(height, width)
    return np.hypot(xx - x, yy - y)


@register("sun-flare", "weather", "Sun flare: a bright glow covering 10-30 %% of the tile", label_safety=True)
def sun_flare(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    radius = random.uniform(0.1, 0.3) * max(height, width)
    distance = radial_distance(height, width, random.uniform(0, width), random.uniform(0, height))
    glow = np.exp(-0.5 * (distance / radius) ** 2) * amount(m, 0.2, 0.7)
    tile.image = blend(tile.image, glow, (255, 250, 235))


@register("vignette", "weather", "Vignetting: corners darkened by 10-50 %%", label_safety=True)
def vignette(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    distance = radial_distance(height, width, random.uniform(0.4, 0.6) * width, random.uniform(0.4, 0.6) * height)
    falloff = (distance / distance.max()) ** 2 * amount(m, 0.1, 0.5)
    tile.image = blend(tile.image, falloff, (0, 0, 0))


@register("spotlight", "weather", "Spotlight: a lit spot, the rest darkened by 10-50 %%", label_safety=True)
def spotlight(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    radius = random.uniform(0.25, 0.6) * max(height, width)
    distance = radial_distance(height, width, random.uniform(0, width), random.uniform(0, height))
    lit = np.exp(-0.5 * (distance / radius) ** 2)
    floor = 1 - amount(m, 0.1, 0.5)
    tile.image = to_uint8(tile.image * (floor + (1 + amount(m, 0, 0.15) - floor) * lit)[..., None])


@register("fog", "weather", "Light fog / haze: 8-35 %% of a light grey veil", label_safety=True)
def fog(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    veil = (0.6 + 0.4 * smooth_noise(height, width, max(height, width) / 2)) * amount(m, 0.08, 0.35)
    tile.image = blend(tile.image, veil, (random.randint(215, 245),) * 3)


@register("wet", "weather", "Wet-surface darkening: irregular patches 12-35 %% darker", label_safety=True)
def wet_surface(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    patches = np.clip((smooth_noise(height, width, random.uniform(48, 128)) - random.uniform(0.35, 0.6)) / 0.15, 0, 1)
    tile.image = blend(tile.image, patches * amount(m, 0.12, 0.35), (0, 0, 0))


# OFF by default. UAV inspection flights are not flown in rain, so rain streaks
# are unrealistic for this imagery, and thin bright streaks are line-shaped
# clutter laid over the cracks.
@register("rain", "weather", "Rain streaks: 30-150 thin slanted light streaks", default_off=True,
          label_safety=True)
def rain(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    angle = math.radians(random.uniform(60, 120))
    streaks = np.zeros((height, width), np.float32)
    for _ in range(int(amount(m, 30, 150))):
        x, y, length = random.uniform(0, width), random.uniform(0, height), random.uniform(8, 20)
        end = (int(x + length * math.cos(angle)), int(y + length * math.sin(angle)))
        cv2.line(streaks, (int(x), int(y)), end, 1.0, 1, cv2.LINE_AA)
    tile.image = blend(tile.image, cv2.GaussianBlur(streaks, (3, 3), 0.6) * random.uniform(0.25, 0.5),
                       (220, 225, 230))


# ----- Blur (after the reference's mild Gaussian / motion blur) -----
# All kernels are at most 5 px. A larger kernel averages a 1-px crack with so
# much background that it disappears while its label remains. Even these small
# ones can erase a faint hairline, so each is followed by the label-safety check.
@register("defocus", "blur", "Defocus blur: disc kernel of radius 0.8-2 px", label_safety=True)
def defocus_blur(tile: Tile, m: float) -> None:
    radius = amount(m, 0.8, 2.0)
    reach = int(math.ceil(radius))
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    kernel = np.clip(radius + 0.5 - np.hypot(xx, yy), 0, 1).astype(np.float32)  # anti-aliased disc
    tile.image = cv2.filter2D(tile.image, -1, kernel / kernel.sum())


@register("median-blur", "blur", "Median blur, 3 px (5 px at high magnitude)", label_safety=True)
def median_blur(tile: Tile, m: float) -> None:
    # A median removes any line narrower than half its kernel, so 3 px already erases 1-px cracks:
    # this operation relies on the label-safety check more than any other.
    tile.image = cv2.medianBlur(np.ascontiguousarray(tile.image), 5 if random.random() < 0.3 * m else 3)


# OFF by default. Zoom blur smears radially and grows with the distance from the
# centre: sharp in the middle, several pixels at the tile border. A hovering or
# cruising UAV does not move toward the surface during an exposure.
@register("zoom-blur", "blur", "Zoom blur: average of zooms up to 3 %%", default_off=True, label_safety=True)
def zoom_blur(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    total = np.zeros(tile.image.shape, np.float32)
    zooms = np.linspace(1.0, 1.0 + amount(m, 0.01, 0.03), 5)
    for zoom in zooms:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, float(zoom))
        total += cv2.warpAffine(tile.image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
    tile.image = to_uint8(total / len(zooms))


# OFF by default. Glass blur swaps neighbouring pixels at random, which breaks a
# 1-px crack into disconnected dots: the opposite of the unbroken centerline
# that clDice and the skeleton pipeline reward.
@register("glass-blur", "blur", "Glass blur: pixels swapped within 1-2 px", default_off=True, label_safety=True)
def glass_blur(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    reach = 2 if m > 0.7 else 1
    yy, xx = pixel_grid(height, width)
    jitter = np.random.randint(-reach, reach + 1, (2, height, width)).astype(np.float32)
    image = cv2.remap(tile.image, xx + jitter[0], yy + jitter[1], cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_REFLECT_101)
    tile.image = cv2.GaussianBlur(image, (3, 3), 0.5)


# ----- Noise and sensor (after the reference's Gaussian noise) -----
@register("poisson-noise", "noise", "Poisson (shot) noise: grows with the square root of the brightness",
          label_safety=True)
def poisson_noise(tile: Tile, m: float) -> None:
    electrons = random.uniform(2.0, 16.0) / max(m, 0.05)  # photo-electrons per grey level: fewer = noisier
    tile.image = to_uint8(np.random.poisson(tile.image.astype(np.float32) * electrons) / electrons)


@register("iso-noise", "noise", "High-ISO noise: luminance noise up to sigma 10 plus blotchy colour noise",
          label_safety=True)
def iso_noise(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    luminance = np.random.normal(0, amount(m, 2.0, 10.0), (height, width, 1))
    # Colour noise is coarser than luminance noise after demosaicing: draw it at half resolution.
    chroma = np.random.normal(0, amount(m, 1.0, 6.0), (height // 2 + 1, width // 2 + 1, 3)).astype(np.float32)
    tile.image = to_uint8(tile.image + luminance + cv2.resize(chroma, (width, height)))


@register("speckle", "noise", "Multiplicative (speckle) noise, sigma 2-10 %%", label_safety=True)
def speckle_noise(tile: Tile, m: float) -> None:
    tile.image = to_uint8(tile.image * (1 + np.random.normal(0, amount(m, 0.02, 0.10), tile.image.shape)))


@register("salt-pepper", "noise", "Salt-and-pepper noise on 0.1-1 %% of the pixels", label_safety=True)
def salt_pepper(tile: Tile, m: float) -> None:
    draw = np.random.rand(*tile.mask.shape)
    share = amount(m, 0.001, 0.01)
    image = tile.image.copy()
    image[draw < share / 2], image[draw > 1 - share / 2] = 0, 255
    tile.image = image


@register("pixel-dropout", "noise", "Pixel dropout: 0.2-2 %% of the pixels set to black", label_safety=True)
def pixel_dropout(tile: Tile, m: float) -> None:
    image = tile.image.copy()
    image[np.random.rand(*tile.mask.shape) < amount(m, 0.002, 0.02)] = 0
    tile.image = image


@register("chromatic-aberration", "noise", "Lateral chromatic aberration: red and blue shifted by up to 1.5 px",
          label_safety=True)
def chromatic_aberration(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    stretch = amount(m, 0.3, 1.5) / (0.5 * math.hypot(height, width))  # 0.3-1.5 px at the tile corner
    image = tile.image.copy()
    for channel, sign in ((0, 1), (2, -1)):  # red is magnified, blue reduced, green stays
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1 + sign * stretch)
        image[..., channel] = cv2.warpAffine(np.ascontiguousarray(tile.image[..., channel]), matrix,
                                             (width, height), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT_101)
    tile.image = image


@register("ringing", "noise", "Ringing / overshoot: a sharp low-pass at 50-85 %% of the Nyquist frequency",
          label_safety=True)
def ringing(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    cutoff = 0.5 * (1 - amount(m, 0.15, 0.5))
    # A brick-wall filter in the frequency domain: its sinc response overshoots at every edge.
    keep = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :]) <= cutoff
    spectrum = np.fft.rfft2(tile.image.astype(np.float32), axes=(0, 1)) * keep[..., None]
    tile.image = to_uint8(np.fft.irfft2(spectrum, s=(height, width), axes=(0, 1)))


# ----- Compression and resolution (after the reference's JPEG) -----
@register("webp", "compression", "WebP compression at quality 30-90", label_safety=True)
def webp_compression(tile: Tile, m: float) -> None:
    quality = int(round(90 - amount(m, 0, 60)))
    try:
        ok, encoded = cv2.imencode(".webp", cv2.cvtColor(tile.image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_WEBP_QUALITY, quality])
    except cv2.error:  # OpenCV built without WebP
        return
    if ok:
        tile.image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@register("downscale", "compression", "Downscale to 50-90 %% and back up: a softer sensor (GSD unchanged)",
          label_safety=True)
def downscale_upscale(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    factor = 1.0 - amount(m, 0.1, 0.5)
    small = cv2.resize(tile.image, (max(8, round(width * factor)), max(8, round(height * factor))),
                       interpolation=cv2.INTER_AREA)
    # The image returns to its own pixel grid: one pixel still covers the same ground and a crack
    # is as many pixels wide as before, only softer. The GSD is a property of the grid, not of the
    # sharpness, so it is NOT updated; telling the model a coarser GSD would be a wrong scale.
    tile.image = cv2.resize(small, (width, height), interpolation=random.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC)))


# ----- Occlusion (after the reference's cutout) -----
# Like cutout, each of these sets valid = 0 on what it hides: a crack label
# under an erased patch would ask the model for something it cannot see.
def hide(tile: Tile, hidden: np.ndarray) -> None:
    image = tile.image.copy()
    image[hidden] = tile.image.mean(axis=(0, 1)).astype(np.uint8)
    tile.image, tile.valid = image, tile.valid * ~hidden


@register("random-erasing", "occlusion", "Random erasing: 2-6 boxes of 3-12 %% of the tile side")
def random_erasing(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    hidden = np.zeros((height, width), bool)
    for _ in range(random.randint(2, 6)):
        box_h, box_w = max(4, int(amount(m, 0.03, 0.12) * height)), max(4, int(amount(m, 0.03, 0.12) * width))
        y, x = random.randint(0, height - box_h), random.randint(0, width - box_w)
        hidden[y:y + box_h, x:x + box_w] = True
    hide(tile, hidden)


@register("gridmask", "occlusion", "GridMask: a regular grid of square holes, 20-50 %% of a 32-96 px period")
def gridmask(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    period = random.randint(32, 96)
    hole = max(4, int(period * amount(m, 0.2, 0.5)))
    yy, xx = np.mgrid[0:height, 0:width]
    hide(tile, ((yy + random.randrange(period)) % period < hole) & ((xx + random.randrange(period)) % period < hole))


@register("hide-and-seek", "occlusion", "Hide-and-seek: 5-30 %% of the cells of a 4, 6 or 8 cell grid hidden")
def hide_and_seek(tile: Tile, m: float) -> None:
    height, width = tile.mask.shape
    cells = random.choice((4, 6, 8))
    hidden = np.random.rand(cells, cells) < amount(m, 0.05, 0.3)
    hidden[random.randrange(cells), random.randrange(cells)] = True  # at least one cell
    hide(tile, cv2.resize(hidden.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0)


# ----- Pipeline order and presets -----
STAGE_OPERATIONS = {stage: [op for op in AUGMENTATIONS.values() if op.stage == stage] for stage in STAGES}


def stage_names(*stages: str) -> list[str]:
    return [op.name for stage in stages for op in STAGE_OPERATIONS[stage]]


# RandAugment pool: pixel-level operations that are safe enough to be on by default.
RANDAUGMENT_POOL = [name for name in stage_names(*RANDAUGMENT_STAGES) if not AUGMENTATIONS[name].default_off]
# Every augmentation in the order it is applied; the names without a stage are the reference's.
# scale -> (crop) -> mixing -> synthetic cracks / distractors -> geometric -> orthomosaic ->
# photometric -> weather -> blur -> noise -> compression -> occlusion.
PIPELINE_ORDER = (
    "rescale", *stage_names("mixing"), "copy-paste",
    "synthetic-hairline", *stage_names("cracks"), "fade", *stage_names("distractors"),
    "dihedral", *stage_names("rescaling"), "affine", *stage_names("warp"), "elastic",
    *stage_names("ortho"),
    "reference-photometric", *stage_names("photometric", "weather", "blur", "noise", "compression"),
    "cutout", *stage_names("occlusion"),
)

# Probability of each operation in the light, medium and heavy presets (the table of the module
# docstring; --self-test checks that the two agree). Operations not listed are off in all three.
PRESET_NAMES = ("reference", "none", "light", "medium", "heavy", "all")
PRESET_PROBABILITIES = {
    "mosaic": (0, 0.05, 0.10), "cutmix": (0, 0.10, 0.15),
    "branch-crack": (0.04, 0.06, 0.08), "varwidth-crack": (0.03, 0.05, 0.06),
    "joint": (0.05, 0.08, 0.10), "sawcut": (0, 0.04, 0.06), "scratch": (0, 0.04, 0.06),
    "painted-line": (0, 0.03, 0.05), "cable-shadow": (0, 0.05, 0.08), "stain": (0.05, 0.08, 0.10),
    "efflorescence": (0, 0.05, 0.08), "dirt": (0, 0.05, 0.08), "spall": (0, 0.04, 0.06),
    "resized-crop": (0, 0.15, 0.25), "zoom-out": (0, 0.08, 0.12),
    "shear": (0.10, 0.15, 0.20), "translate": (0.10, 0.15, 0.20), "perspective": (0.10, 0.15, 0.20),
    "optical": (0, 0.08, 0.12), "grid-distortion": (0, 0.05, 0.10),
    "seam": (0.05, 0.08, 0.12), "nodata-border": (0, 0.03, 0.05), "exposure-step": (0.05, 0.08, 0.12),
    "blur-patch": (0, 0.05, 0.08),
    "rgb-shift": (0.10, 0.12, 0.15), "channel-shuffle": (0, 0, 0.02), "color-jitter": (0.15, 0.20, 0.25),
    "white-balance": (0.15, 0.20, 0.25), "tone-curve": (0.10, 0.15, 0.20), "autocontrast": (0, 0.05, 0.08),
    "equalize": (0, 0, 0.05), "sharpen": (0.05, 0.08, 0.10), "emboss": (0, 0, 0.02), "posterize": (0, 0, 0.02),
    "overexposure": (0, 0.04, 0.06), "underexposure": (0, 0.04, 0.06),
    "shadow": (0.08, 0.12, 0.15), "sun-flare": (0, 0, 0.03), "vignette": (0.05, 0.08, 0.10),
    "spotlight": (0, 0.05, 0.08), "fog": (0, 0, 0.03), "wet": (0, 0.05, 0.08),
    "defocus": (0.03, 0.05, 0.08), "median-blur": (0, 0.02, 0.04),
    "poisson-noise": (0.08, 0.10, 0.15), "iso-noise": (0.05, 0.08, 0.10), "speckle": (0, 0.05, 0.08),
    "salt-pepper": (0, 0, 0.03), "pixel-dropout": (0, 0.03, 0.05), "chromatic-aberration": (0, 0.05, 0.08),
    "ringing": (0, 0.03, 0.05),
    "webp": (0.08, 0.10, 0.15), "downscale": (0.08, 0.10, 0.15),
    "random-erasing": (0.05, 0.08, 0.10), "gridmask": (0, 0.04, 0.06), "hide-and-seek": (0, 0.04, 0.06),
}
ALL_PRESET_PROBABILITY = 0.05
PRESET_MAGNITUDE = {"reference": 1.0, "none": 1.0, "light": 0.5, "medium": 0.75, "heavy": 1.0, "all": 0.75}
PRESET_MAX_ROTATION = {"medium": 45.0, "heavy": 180.0, "all": 180.0}
# "none" switches the reference augmentation off as well.
NO_AUGMENTATION = {"scale_p": 0.0, "copy_paste_p": 0.0, "synthetic_p": 0.0, "fade_p": 0.0, "affine_p": 0.0,
                   "elastic_p": 0.0, "cutout_p": 0.0, "contrast_mode": "off"}
assert not set(PRESET_PROBABILITIES) - set(AUGMENTATIONS)
assert not any(AUGMENTATIONS[name].default_off for name in PRESET_PROBABILITIES)


def preset_probabilities(preset: str) -> dict[str, float]:
    """Probability of every registered operation in a preset."""
    if preset == "all":
        return {name: ALL_PRESET_PROBABILITY for name in AUGMENTATIONS}
    if preset in ("light", "medium", "heavy"):
        column = ("light", "medium", "heavy").index(preset)
        return {name: float(PRESET_PROBABILITIES.get(name, (0, 0, 0))[column]) for name in AUGMENTATIONS}
    return {name: 0.0 for name in AUGMENTATIONS}


class CrackSegmentationDataset(Dataset):
    """Returns (image, mask, valid, domain_id, index, gsd); gsd is in mm/px, NaN when unknown."""

    def __init__(
        self, pairs, domain_ids, gsds=None, training: bool = False, crop_size: int = 256,
        options: AugmentationOptions | None = None,
    ):
        if not pairs:
            raise ValueError("Empty dataset split.")
        self.pairs, self.domain_ids = pairs, domain_ids
        self.gsds = [UNKNOWN_GSD] * len(pairs) if gsds is None else [float(gsd) for gsd in gsds]
        self.training, self.crop_size = training, crop_size
        self.options = options or AugmentationOptions()
        # Mixing partners (copy-paste donors, CutMix, MixUp, mosaic) come from the same source, so
        # label width conventions (4 px DXF lines vs 18 px CRACK500 regions) are never mixed in
        # one image, and from the same GSD (or all unknown), so that one tile never shows two
        # ground resolutions while the network is told a single GSD for it.
        self.groups = [(domain, gsd_key(gsd)) for domain, gsd in zip(domain_ids, self.gsds)]
        self.members = defaultdict(list)
        for index, group in enumerate(self.groups):
            self.members[group].append(index)
        # Evaluation batches group equal image sizes, so read (h, w) headers once.
        self.shapes = None if training else [image_shape(mask) for _, mask in pairs]
        # "all" contrast mode: each image steps through every level as it is drawn.
        self.contrast_factors = None
        if training and self.options.contrast_mode == "all":
            self.contrast_factors = contrast_level_factors(self.options)
            levels, generator = len(self.contrast_factors), np.random.default_rng(2026)
            # One shuffled level order for all images, entered at a different point by
            # each image, so every epoch and batch holds a balanced mix of levels.
            self.level_order = generator.permutation(levels).tolist()
            self.level_offsets = (generator.permutation(len(pairs)) % levels).tolist()
            # Draw counters in shared memory, so DataLoader workers update the same arrays.
            self.image_draws = torch.zeros(len(pairs), dtype=torch.int64).share_memory_()
            self.level_draws = torch.zeros(levels, dtype=torch.int64).share_memory_()
        # (probability, magnitude) of every registered operation; all off unless the options say otherwise.
        self.settings = {name: tuple(self.options.ops.get(name, (0.0, 1.0))) for name in AUGMENTATIONS}
        # How often each augmentation fired, and what the label-safety check did:
        # [samples drawn, checks, crack pixels checked, crack pixels excluded]. Shared memory, as above.
        self.counter_index = {name: position for position, name in enumerate(PIPELINE_ORDER)}
        self.fired = torch.zeros(len(PIPELINE_ORDER), dtype=torch.int64).share_memory_()
        self.safety = torch.zeros(4, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.pairs)

    def next_contrast_factor(self, index: int) -> float | None:
        """Contrast factor for this draw of image `index`; None in "random" mode."""
        if self.options.contrast_mode == "off":
            return 1.0  # the original contrast, and no random change either
        if self.contrast_factors is None:
            return None
        position = (self.level_offsets[index] + int(self.image_draws[index])) % len(self.contrast_factors)
        level = self.level_order[position]
        self.image_draws[index] += 1
        self.level_draws[level] += 1
        return self.contrast_factors[level]

    def contrast_coverage(self) -> dict | None:
        """How completely training has covered the contrast levels so far."""
        if self.contrast_factors is None:
            return None
        draws, levels = self.image_draws.clone(), len(self.contrast_factors)
        return {
            "mode": "all", "levels": levels,
            "contrast_change_percent": [round(100 * (factor - 1), 3) for factor in self.contrast_factors],
            "draws_per_level": self.level_draws.tolist(),
            "training_images": len(self.pairs),
            "images_trained_at_every_level": int((draws >= levels).sum()),
            "minimum_draws_per_image": int(draws.min()), "mean_draws_per_image": float(draws.float().mean()),
        }

    def augmentation_counts(self) -> dict:
        """How often each augmentation fired and what the label-safety check excluded so far."""
        samples, checks, checked, excluded = self.safety.tolist()
        fired = dict(zip(PIPELINE_ORDER, self.fired.tolist()))
        return {
            "training_samples_drawn": samples,
            "fired": fired,
            "fired_share_of_samples": {name: round(count / max(1, samples), 5) for name, count in fired.items()},
            "label_safety": {
                "enabled": bool(self.options.label_safety), "threshold_grey_levels": LABEL_SAFETY_GREY_LEVELS,
                "checks": checks, "crack_pixels_checked": checked, "crack_pixels_excluded": excluded,
            },
        }

    def load_partner(self, index: int, scale: float = 1.0, focus_p: float = 1.0):
        """Image, mask and valid crop of a tile that may be mixed into tile `index`.

        It comes from the same source family and the same GSD (see self.groups), and is
        resampled by `scale`, the scale tile `index` has reached, so that the two still show
        the same GSD when they are combined.
        """
        image, mask = read_pair(*self.pairs[random.choice(self.members[self.groups[index]])])
        valid = np.ones_like(mask)
        if scale == 1.0:
            return random_crop(image, mask, valid, self.crop_size, focus_p)
        # Crop the source window that `scale` maps onto one crop and resample only that window,
        # instead of resampling a whole 12-megapixel photo.
        window = max(32, int(round(self.crop_size / scale)))
        return resize_tile(*random_crop(image, mask, valid, window, focus_p), self.crop_size, self.crop_size)

    def paste_donor_crack(self, image, mask, index: int, scale: float = 1.0):
        """Transfer another sample's crack *contrast* (dark or light), not its texture patch."""
        donor_image, donor_mask, donor_valid = self.load_partner(index, scale, focus_p=1.0)
        donor_mask = donor_mask * donor_valid
        if not donor_mask.any():
            return image, mask
        difference = donor_image.astype(np.float32) - local_background(donor_image, 15).astype(np.float32)
        if abs(difference[donor_mask > 0].mean()) < 4.0:
            return image, mask  # no visible contrast: pasting it would teach invisible labels
        region = cv2.dilate(donor_mask, np.ones((3, 3), np.uint8)).astype(np.float32)
        alpha = cv2.GaussianBlur(region, (5, 5), 0)[..., None]
        blended = image.astype(np.float32) + alpha * difference
        return np.clip(blended, 0, 255).astype(np.uint8), np.maximum(mask, donor_mask)

    # ----- Extended library -----
    def count(self, name: str) -> None:
        self.fired[self.counter_index[name]] += 1

    def run(self, operation: Augmentation, tile: Tile, magnitude: float) -> None:
        if operation.label_safety and self.options.label_safety and tile.before is None and tile.mask.any():
            tile.before = tile.image.copy()  # opens a label-safety comparison; check_labels() closes it
        operation.function(tile, magnitude)

    def run_stage(self, stage: str, tile: Tile, forced: dict) -> None:
        """Apply the stage's operations that fire: by their own probability, or forced by RandAugment."""
        for operation in STAGE_OPERATIONS[stage]:
            probability, magnitude = self.settings[operation.name]
            if operation.name in forced:
                magnitude = forced[operation.name]
            elif probability <= 0 or random.random() >= probability:  # an operation that is off draws nothing
                continue
            self.run(operation, tile, magnitude)
            self.count(operation.name)
            if stage == "mixing":
                # One mix per tile: a mosaic of CutMix tiles is more patchwork than image.
                break

    def check_labels(self, tile: Tile, count: bool = True) -> None:
        """Close the label-safety comparison: exclude the cracks made invisible since tile.before."""
        if tile.before is None:
            return
        tile.valid, excluded, checked = label_safety_check(tile.before, tile.image, tile.mask, tile.valid)
        tile.before = None
        if count:
            self.safety[1:] += torch.tensor((1, checked, excluded))

    def augment(self, tile: Tile, index: int) -> Tile:
        """The training pipeline. The reference steps are in the reference's order and draw the same
        random numbers; with the "reference" preset nothing else runs, so the augmentation is the
        reference's. The run_stage() calls add the extended library in between."""
        options = self.options
        forced = {}
        if options.randaugment:  # N random pixel-level operations at magnitude M / 10
            count, magnitude = options.randaugment
            forced = {name: magnitude / 10.0
                      for name in random.sample(RANDAUGMENT_POOL, min(int(count), len(RANDAUGMENT_POOL)))}
        tile.partner = lambda: self.load_partner(index, tile.scale, options.focus_p)
        tile.label_width = options.synthetic_label_width
        # 1. Scale. Enlarging by s makes every crack s times wider in pixels: GSD / s.
        if random.random() < options.scale_p:
            tile.image, tile.mask, tile.valid, factor = rescale(tile.image, tile.mask, tile.valid,
                                                                random.uniform(0.75, 1.25))
            tile.scale *= factor
            self.count("rescale")
        # 2. Crop. A crop shows fewer pixels of the same size: the GSD does not change.
        tile.image, tile.mask, tile.valid = random_crop(tile.image, tile.mask, tile.valid, self.crop_size,
                                                        options.focus_p)
        # 3. Mixing. Partners and donors are brought to tile.scale, which the mosaic halves.
        self.run_stage("mixing", tile, forced)
        if random.random() < options.copy_paste_p:
            tile.image, tile.mask = self.paste_donor_crack(tile.image, tile.mask, index, tile.scale)
            self.count("copy-paste")
        # 4. Synthetic cracks, then fading (which also fades them), then background distractors.
        image, mask = add_synthetic_hairline(tile.image, tile.mask, options.synthetic_p,
                                             options.synthetic_label_width)
        if image is not tile.image:
            self.count("synthetic-hairline")
        tile.image, tile.mask = image, mask
        self.run_stage("cracks", tile, forced)
        image = fade_cracks(tile.image, tile.mask, options.fade_p)
        if image is not tile.image:
            self.count("fade")
        tile.image = image
        self.run_stage("distractors", tile, forced)
        self.check_labels(tile)  # before the geometry changes: the comparison needs aligned images
        # 5. Geometric. Flips and 90-degree turns keep the pixel size; the others report their scale.
        if options.dihedral:
            tile.image, tile.mask, tile.valid = dihedral(tile.image, tile.mask, tile.valid)
            self.count("dihedral")
        self.run_stage("rescaling", tile, forced)
        if random.random() < options.affine_p:
            tile.image, tile.mask, tile.valid, factor = random_affine(tile.image, tile.mask, tile.valid,
                                                                      options.max_rotation)
            tile.scale *= factor
            self.count("affine")
        self.run_stage("warp", tile, forced)
        if random.random() < options.elastic_p:
            tile.image, tile.mask, tile.valid = elastic(tile.image, tile.mask, tile.valid)
            self.count("elastic")
        # 6. Orthomosaic artefacts.
        self.run_stage("ortho", tile, forced)
        # 7-11. Photometric, weather, blur, noise, compression: the reference chain stage by stage,
        # each followed by the new operations of that stage. None of them changes the pixel grid.
        if options.photometric:
            tile.image = reference_tone(tile.image, options, self.next_contrast_factor(index))
            self.count("reference-photometric")
        self.run_stage("photometric", tile, forced)
        self.run_stage("weather", tile, forced)
        if options.photometric:
            tile.image = reference_blur(tile.image)
        self.run_stage("blur", tile, forced)
        if options.photometric:
            tile.image = reference_noise(tile.image)
        self.run_stage("noise", tile, forced)
        if options.photometric:
            tile.image = reference_jpeg(tile.image)
        self.run_stage("compression", tile, forced)
        self.check_labels(tile)  # one check for everything since the first contrast-reducing operation
        # 12. Occlusion: hidden regions leave the loss.
        if random.random() < options.cutout_p:
            tile.image, tile.valid = cutout(tile.image, tile.valid)
            self.count("cutout")
        self.run_stage("occlusion", tile, forced)
        tile.partner = None  # the closure refers back to the tile: drop it so the arrays are freed at once
        return tile

    def enabled_augmentations(self) -> list[str]:
        """Names of the augmentations that can fire with the current options, in pipeline order."""
        options = self.options
        reference = {
            "rescale": options.scale_p, "copy-paste": options.copy_paste_p, "synthetic-hairline": options.synthetic_p,
            "fade": options.fade_p, "dihedral": float(options.dihedral), "affine": options.affine_p,
            "elastic": options.elastic_p, "reference-photometric": float(options.photometric),
            "cutout": options.cutout_p,
        }
        pool = set(RANDAUGMENT_POOL) if options.randaugment and options.randaugment[0] > 0 else set()
        return [name for name in PIPELINE_ORDER
                if reference.get(name, 0) > 0 or name in pool or self.settings.get(name, (0, 0))[0] > 0]

    def apply_alone(self, name: str, tile: Tile, index: int) -> Tile:
        """One augmentation, forced, on a copy of `tile`: for the preview and the self-tests."""
        options, tile = self.options, tile.copy()
        tile.partner = lambda: self.load_partner(index, tile.scale, options.focus_p)
        tile.label_width = options.synthetic_label_width
        if name in AUGMENTATIONS:
            probability, magnitude = self.settings[name]
            if probability <= 0 and options.randaugment:
                magnitude = options.randaugment[1] / 10.0
            self.run(AUGMENTATIONS[name], tile, magnitude)
            self.check_labels(tile, count=False)
        elif name == "rescale":
            image, mask, valid, factor = rescale(tile.image, tile.mask, tile.valid, random.uniform(0.75, 1.25))
            tile.image, tile.mask, tile.valid = random_crop(image, mask, valid, tile.mask.shape[0], 1.0)
            tile.scale *= factor
        elif name == "copy-paste":
            tile.image, tile.mask = self.paste_donor_crack(tile.image, tile.mask, index, tile.scale)
        elif name == "synthetic-hairline":
            tile.image, tile.mask = add_synthetic_hairline(tile.image, tile.mask, 1.0, options.synthetic_label_width)
        elif name == "fade":
            tile.image = fade_cracks(tile.image, tile.mask, 1.0)
        elif name == "dihedral":
            tile.image, tile.mask, tile.valid = dihedral(tile.image, tile.mask, tile.valid)
        elif name == "affine":
            tile.image, tile.mask, tile.valid, factor = random_affine(tile.image, tile.mask, tile.valid,
                                                                      options.max_rotation)
            tile.scale *= factor
        elif name == "elastic":
            tile.image, tile.mask, tile.valid = elastic(tile.image, tile.mask, tile.valid)
        elif name == "reference-photometric":
            factor = random.choice(self.contrast_factors) if self.contrast_factors else None
            tile.image = photometric_augment(tile.image, options, 1.0 if options.contrast_mode == "off" else factor)
        elif name == "cutout":
            tile.image, tile.valid = cutout(tile.image, tile.valid)
        else:
            raise KeyError(name)
        tile.partner = None
        return tile

    def __getitem__(self, index):
        image, mask = read_pair(*self.pairs[index])
        valid = np.ones_like(mask)
        gsd = self.gsds[index]
        if self.training:
            self.safety[0] += 1
            tile = self.augment(Tile(image, mask, valid), index)
            image, mask, valid = tile.image, tile.mask, tile.valid
            # The GSD follows the accumulated scale change. NaN (unknown) divided by anything stays NaN.
            gsd = gsd / tile.scale
        else:
            # Full image at native resolution, padded bottom/right to a multiple of 32.
            height, width = mask.shape
            pad_h, pad_w = (-height) % 32, (-width) % 32
            if pad_h or pad_w:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                valid = cv2.copyMakeBorder(valid, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)[None])
        valid_tensor = torch.from_numpy((valid > 0).astype(np.float32)[None])
        return (image_tensor, mask_tensor * valid_tensor, valid_tensor, self.domain_ids[index], index,
                torch.tensor(gsd, dtype=torch.float32))


# --------------------------------------------------------------------------- #
# Augmentation preview
# --------------------------------------------------------------------------- #
def preview_panel(tile: Tile, title: str) -> np.ndarray:
    """image | label overlay | valid map, under a title line.

    Overlay: red = supervised crack label, blue = crack label taken out of the loss.
    Valid map: white = supervised, dark = excluded from the loss.
    """
    image = np.ascontiguousarray(tile.image)
    overlay = image.astype(np.float32)
    for region, colour in (((tile.mask > 0) & (tile.valid > 0), (255, 40, 40)),
                           ((tile.mask > 0) & (tile.valid == 0), (40, 90, 255))):
        overlay[region] = 0.35 * overlay[region] + 0.65 * np.float32(colour)
    valid = np.repeat((tile.valid > 0).astype(np.uint8)[..., None] * 215 + 40, 3, axis=2)
    panel = np.hstack((image, overlay.astype(np.uint8), valid))
    header = np.full((22, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(header, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (11, 11, 11), 1, cv2.LINE_AA)
    return np.vstack((header, panel))


def save_augmentation_preview(dataset: CrackSegmentationDataset, folder: Path, count: int, seed: int) -> list[str]:
    """For `count` training tiles, write one panel per enabled augmentation, applied alone, and an
    overview sheet of all of them. Returns the augmentations shown."""
    folder.mkdir(parents=True, exist_ok=True)
    random_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    names = dataset.enabled_augmentations()
    # Tiles with cracks show what happens to the labels; fall back to any tile if none are found.
    order = random.Random(seed).sample(range(len(dataset)), len(dataset))
    chosen = [index for index in order[:25 * count]
              if cv2.imread(str(dataset.pairs[index][1]), cv2.IMREAD_GRAYSCALE).any()][:count] or order[:count]
    for index in chosen:
        image, mask = read_pair(*dataset.pairs[index])
        base = Tile(*(np.ascontiguousarray(part) for part in
                      random_crop(image, mask, np.ones_like(mask), dataset.crop_size, 1.0)))
        gsd, stem = dataset.gsds[index], dataset.pairs[index][0].stem
        panels = [preview_panel(base, f"original | GSD {format_gsd(gsd)} mm/px")]
        for position, name in enumerate(names, start=1):
            tile = dataset.apply_alone(name, base, index)
            panels.append(preview_panel(
                tile, f"{name} | GSD {format_gsd(gsd)} -> {format_gsd(gsd / tile.scale)} mm/px"))
            cv2.imwrite(str(folder / f"{stem}__{position:02d}_{name}.png"),
                        cv2.cvtColor(panels[-1], cv2.COLOR_RGB2BGR))
        columns = 1 if len(panels) <= 6 else 2 if len(panels) <= 24 else 3
        panels += [np.full_like(panels[0], 255)] * ((-len(panels)) % columns)
        sheet = np.vstack([np.hstack(panels[start:start + columns]) for start in range(0, len(panels), columns)])
        cv2.imwrite(str(folder / f"{stem}__overview.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    # The preview must not shift the random stream of the training run that follows.
    random.setstate(random_state)
    np.random.set_state(numpy_state)
    return names


def shape_grouped_batches(shapes, batch_size: int, indices=None, max_pixels: float = math.inf) -> list[list[int]]:
    """Batches of equally sized images; `indices` restricts them to a subset of the dataset.

    A batch holds at most `batch_size` images and `max_pixels` pixels, so a
    full-resolution photo is evaluated on its own.
    """
    groups = defaultdict(list)
    for index in (range(len(shapes)) if indices is None else indices):
        height, width = shapes[index]
        groups[(height + (-height) % 32, width + (-width) % 32)].append(index)
    batches = []
    for (height, width), members in sorted(groups.items()):
        size = int(max(1, min(batch_size, max_pixels // (height * width))))
        batches.extend(members[start:start + size] for start in range(0, len(members), size))
    return batches


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(ConvBNReLU(input_channels, output_channels), ConvBNReLU(output_channels, output_channels))


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.conv = DoubleConv(input_channels + skip_channels, output_channels)
        self.attention = SCSE(output_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.attention(self.conv(torch.cat((x, skip), 1)))


class ThinLineResponse(nn.Module):
    """Fixed morphological black-hat and white top-hat on grayscale input.

    Black-hat (closing - image) responds to dark structures narrower than the
    kernel; top-hat (image - opening) to bright ones. Both are needed: in EH104
    many cracks appear lighter than the concrete (deposits in the crack), not
    darker. The module has no parameters, so it is a thin-line prior at no data cost.
    """

    def __init__(self, kernel_sizes=(7, 15)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.output_channels = 2 * len(self.kernel_sizes)
        self.register_buffer("luma", torch.tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1))

    def forward(self, rgb):
        gray = (rgb * self.luma).sum(1, keepdim=True)
        responses = []
        for size in self.kernel_sizes:
            closed = -F.max_pool2d(-F.max_pool2d(gray, size, 1, size // 2), size, 1, size // 2)
            opened = F.max_pool2d(-F.max_pool2d(-gray, size, 1, size // 2), size, 1, size // 2)
            responses.extend((closed - gray, gray - opened))
        return torch.cat(responses, 1)


class FiLMGenerator(nn.Module):
    """Turns a tile's GSD into a per-channel scale and shift for every conditioned feature map.

    normalized log(GSD) -> Linear(1, hidden) -> SiLU -> Linear(hidden, hidden) -> SiLU, then one
    Linear(hidden, 2C) per layer gives (d_gamma, beta) for F' = (1 + d_gamma) * F + beta.
    """

    def __init__(self, layer_channels: dict[str, int], hidden: int = 64):
        super().__init__()
        self.embed = nn.Linear(1, hidden)
        self.trunk = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 2 * channels)
                                    for name, channels in layer_channels.items()})
        # Zero heads give d_gamma = 0 and beta = 0, so F' = F: the network starts as the plain
        # U-Net with its pretrained encoder undisturbed, and uses the GSD only as far as the
        # gradient finds it useful.
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Stands in for embed(gsd) when the GSD is unknown. It starts as the embedding of the mean
        # training GSD (normalized input 0, so embed gives its bias): the best guess for a tile
        # nothing is known about.
        self.null_embedding = nn.Parameter(self.embed.bias.detach().clone())

    def forward(self, normalized_log_gsd: torch.Tensor, known: torch.Tensor) -> dict:
        embedding = self.embed(normalized_log_gsd[:, None])
        embedding = torch.where(known[:, None], embedding, self.null_embedding.to(embedding.dtype))
        hidden = self.trunk(embedding)
        return {name: head(hidden).view(len(hidden), 2, -1, 1, 1).unbind(1) for name, head in self.heads.items()}


class HairlineFiLMUNet(nn.Module):
    """ResNet-encoder U-Net with a full-resolution skip for 1-4 px cracks, conditioned on the GSD.

    Input: RGB in [0, 1] and `gsd`, a float tensor (N,) in mm/px with NaN (or None) for unknown.
    Eval mode returns logits (N, 1, H, W). Train mode with deep supervision returns
    (logits, [auxiliary logits]).

    ``use_film=False`` gives the same network without the conditioning (it ignores `gsd`): the
    baseline of the FiLM ablation.
    """

    def __init__(
        self, encoder_name: str = "resnet34", pretrained: bool = True,
        decoder_channels=(256, 128, 64, 48, 32), stem_channels: int = 32,
        line_prior: bool = True, deep_supervision: bool = True, prior_probability: float = 0.03,
        use_film: bool = True, film_skips: bool = False, film_hidden: int = 64, gsd_dropout: float = 0.15,
        gsd_log_mean: float = 0.0, gsd_log_std: float = 1.0,
    ):
        super().__init__()
        try:
            import torchvision
        except ImportError as exc:
            raise RuntimeError("The encoder requires torchvision: python -m pip install torchvision") from exc
        try:
            self.encoder = torchvision.models.get_model(encoder_name, weights="DEFAULT" if pretrained else None)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load ImageNet {encoder_name} weights. In Kaggle, enable Internet "
                    "for the first run, or pass --no-pretrained."
                ) from exc
            raise
        self.encoder.fc = nn.Identity()
        c0, c1, c2, c3, c4 = ENCODER_CHANNELS[encoder_name]
        d4, d3, d2, d1, d0 = decoder_channels
        self.thin_lines = ThinLineResponse() if line_prior else None
        stem_inputs = 3 + (self.thin_lines.output_channels if line_prior else 0)
        self.full_resolution = DoubleConv(stem_inputs, stem_channels)
        self.dec4 = DecoderBlock(c4, c3, d4)              # 1/16
        self.dec3 = DecoderBlock(d4, c2, d3)              # 1/8
        self.dec2 = DecoderBlock(d3, c1, d2)              # 1/4
        self.dec1 = DecoderBlock(d2, c0, d1)              # 1/2
        self.dec0 = DecoderBlock(d1, stem_channels, d0)   # 1/1
        self.head = nn.Conv2d(d0, 1, 1)
        self.deep_supervision = deep_supervision
        # Auxiliary heads on the decoder outputs at 1/8 and 1/4 scale.
        self.auxiliary_heads = nn.ModuleList([nn.Conv2d(d3, 1, 1), nn.Conv2d(d2, 1, 1)]) if deep_supervision else None
        # Start near the crack prior so early training is not dominated by background.
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head, *(self.auxiliary_heads or [])]:
            nn.init.constant_(head.bias, bias)
        # FiLM: one (d_gamma, beta) pair per decoder block output, and per skip with film_skips.
        self.film = None
        if use_film:
            layers = {"dec4": d4, "dec3": d3, "dec2": d2, "dec1": d1, "dec0": d0}
            if film_skips:
                layers.update({"skip4": c3, "skip3": c2, "skip2": c1, "skip1": c0, "skip0": stem_channels})
            self.film = FiLMGenerator(layers, film_hidden)
        self.gsd_dropout = gsd_dropout
        # Plain numbers, not buffers: they are fixed statistics of the training set, and the EMA
        # would otherwise average them like weights.
        self.gsd_log_mean, self.gsd_log_std = float(gsd_log_mean), float(gsd_log_std)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def film_parameters(self, gsd, batch: int, device) -> dict:
        """{layer: (d_gamma, beta)} for the batch; empty without FiLM."""
        if self.film is None:
            return {}
        if gsd is None:
            gsd = float("nan")
        gsd = torch.as_tensor(gsd, dtype=torch.float32, device=device).reshape(-1)
        if gsd.numel() == 1:
            gsd = gsd.expand(batch)
        known = torch.isfinite(gsd) & (gsd > 0)
        if self.training and self.gsd_dropout > 0:
            # GSD dropout: some known tiles are trained as "unknown", so the null embedding is
            # trained on every source and the model does not come to depend on being told the GSD.
            known = known & (torch.rand(batch, device=device) >= self.gsd_dropout)
        # log(GSD), standardised with the training-set statistics. Unknown entries get a dummy 1
        # so the logarithm stays finite; their embedding is replaced by the null embedding.
        normalized = (torch.log(torch.where(known, gsd, torch.ones_like(gsd))) - self.gsd_log_mean) / self.gsd_log_std
        return self.film(normalized, known)

    @staticmethod
    def modulate(features: torch.Tensor, film: dict, layer: str) -> torch.Tensor:
        """F' = (1 + d_gamma) * F + beta; the identity when `layer` is not conditioned."""
        if layer not in film:
            return features
        delta_gamma, beta = film[layer]
        return features * (1 + delta_gamma) + beta

    def forward(self, x, gsd=None):
        size = x.shape[-2:]
        film = self.film_parameters(gsd, x.size(0), x.device)
        modulate = self.modulate
        normalized = (x - self.image_mean) / self.image_std
        stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
        full = self.full_resolution(stem_input)
        encoder = self.encoder
        e0 = encoder.relu(encoder.bn1(encoder.conv1(normalized)))   # 1/2
        e1 = encoder.layer1(encoder.maxpool(e0))                     # 1/4
        e2 = encoder.layer2(e1)                                      # 1/8
        e3 = encoder.layer3(e2)                                      # 1/16
        e4 = encoder.layer4(e3)                                      # 1/32
        d4 = modulate(self.dec4(e4, modulate(e3, film, "skip4")), film, "dec4")      # 1/16
        d3 = modulate(self.dec3(d4, modulate(e2, film, "skip3")), film, "dec3")      # 1/8
        d2 = modulate(self.dec2(d3, modulate(e1, film, "skip2")), film, "dec2")      # 1/4
        d1 = modulate(self.dec1(d2, modulate(e0, film, "skip1")), film, "dec1")      # 1/2
        d0 = modulate(self.dec0(d1, modulate(full, film, "skip0")), film, "dec0")    # 1/1
        logits = self.head(d0)
        if self.training and self.deep_supervision:
            auxiliary = [
                F.interpolate(head(features), size=size, mode="bilinear", align_corners=False)
                for head, features in zip(self.auxiliary_heads, (d3, d2))
            ]
            return logits, auxiliary
        return logits


def load_model_from_checkpoint(path, device="cpu"):
    """Rebuild a HairlineFiLMUNet from best.pt for inference scripts.

    The architecture kwargs in the checkpoint carry the FiLM flag and the GSD normalisation, so the
    model is called as ``model(images, gsd)`` with the GSD in mm/px (NaN or None when unknown).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HairlineFiLMUNet(pretrained=False, **checkpoint["architecture"]["kwargs"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


class ModelEMA:
    """Exponential moving average of weights (and BN statistics)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, ramp_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.ramp_steps, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp_steps))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def soft_erode(image):
    return torch.minimum(
        -F.max_pool2d(-image, (3, 1), 1, (1, 0)), -F.max_pool2d(-image, (1, 3), 1, (0, 1))
    )


def soft_skeleton(image, iterations: int):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability, target, iterations: int, smooth: float = 1.0):
    predicted_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    topology_precision = ((predicted_skeleton * target).sum() + smooth) / (predicted_skeleton.sum() + smooth)
    topology_sensitivity = ((target_skeleton * probability).sum() + smooth) / (target_skeleton.sum() + smooth)
    return 1 - 2 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)


class HairlineCrackLoss(nn.Module):
    """BCE + Tversky (region) + clDice (centerline continuity), valid-pixel masked."""

    def __init__(
        self, bce_weight=1.0, tversky_weight=1.0, cldice_weight=0.5,
        false_positive_weight=0.4, false_negative_weight=0.6,
        skeleton_iterations=10, auxiliary_weights=(0.4, 0.2),
    ):
        super().__init__()
        self.bce_weight, self.tversky_weight, self.cldice_weight = bce_weight, tversky_weight, cldice_weight
        self.fp_weight, self.fn_weight = false_positive_weight, false_negative_weight
        self.skeleton_iterations = skeleton_iterations
        self.auxiliary_weights = tuple(auxiliary_weights)

    def region_loss(self, logits, target, valid):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        probability = torch.sigmoid(logits) * valid
        target = target * valid
        true_positive = (probability * target).sum()
        false_positive = (probability * (1 - target)).sum()
        false_negative = ((1 - probability) * target).sum()
        tversky = (true_positive + 1.0) / (
            true_positive + self.fp_weight * false_positive + self.fn_weight * false_negative + 1.0
        )
        return self.bce_weight * bce + self.tversky_weight * (1 - tversky), probability, target

    def forward(self, outputs, target, valid, cldice_scale: float = 1.0):
        logits, auxiliary = outputs if isinstance(outputs, tuple) else (outputs, ())
        loss, probability, target_valid = self.region_loss(logits, target, valid)
        weight = self.cldice_weight * cldice_scale
        if weight > 0:
            loss = loss + weight * soft_cldice_loss(probability, target_valid, self.skeleton_iterations)
        for auxiliary_weight, auxiliary_logits in zip(self.auxiliary_weights, auxiliary):
            loss = loss + auxiliary_weight * self.region_loss(auxiliary_logits, target, valid)[0]
        return loss


class FocalCrackLoss(HairlineCrackLoss):
    """Binary focal loss (Lin et al., 2017) in place of BCE + Tversky, valid-pixel masked.

    The clDice term, its warm-up and the deep-supervision weights are those of
    HairlineCrackLoss, so ``--loss focal`` changes the region term only.
    """

    def __init__(self, alpha=0.25, gamma=2.0, cldice_weight=0.5, skeleton_iterations=10,
                 auxiliary_weights=(0.4, 0.2)):
        super().__init__(cldice_weight=cldice_weight, skeleton_iterations=skeleton_iterations,
                         auxiliary_weights=auxiliary_weights)
        self.alpha, self.gamma = alpha, gamma

    def region_loss(self, logits, target, valid):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        # p_t is the probability given to the true class: (1 - p_t)^gamma turns the loss down on
        # pixels that are already right, which is nearly all of the background.
        true_class = probability * target + (1 - probability) * (1 - target)
        class_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = class_weight * (1 - true_class) ** self.gamma * bce
        focal = (focal * valid).sum() / valid.sum().clamp_min(1.0)
        return focal, probability * valid, target * valid


# --------------------------------------------------------------------------- #
# Metrics: every threshold at once, strict and with an N-pixel tolerance
# --------------------------------------------------------------------------- #
class ThresholdSweep:
    """Accumulate per-source counts for all thresholds in one pass.

    Tolerant precision = predicted pixels within `tolerance` px of a label;
    tolerant recall = label pixels within `tolerance` px of a prediction.
    """

    def __init__(self, thresholds, tolerance: int, domain_count: int, device):
        self.thresholds = [float(value) for value in thresholds]
        self.threshold_tensor = torch.tensor(self.thresholds, device=device).view(-1, 1, 1)
        self.tolerance = tolerance
        self.counts = torch.zeros(domain_count, len(self.thresholds), len(COUNT_NAMES),
                                  dtype=torch.float64, device=device)
        self.pool_dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32

    def reset(self):
        self.counts.zero_()

    def dilate(self, binary):
        if self.tolerance <= 0:
            return binary
        size = 2 * self.tolerance + 1
        return F.max_pool2d(binary[:, None].to(self.pool_dtype), size, 1, self.tolerance)[:, 0] > 0

    @torch.no_grad()
    def update(self, probabilities, targets, valid, domain_ids):
        for probability, target, valid_map, domain in zip(probabilities, targets, valid, domain_ids.tolist()):
            valid_map = valid_map[0] > 0.5
            truth = (target[0] >= 0.5) & valid_map
            prediction = (probability[0].unsqueeze(0) >= self.threshold_tensor) & valid_map
            true_positive = (prediction & truth).sum((1, 2))
            predicted = prediction.sum((1, 2))
            labelled = truth.sum().expand_as(true_positive)
            predicted_near_label = (prediction & self.dilate(truth[None])).sum((1, 2))
            label_near_prediction = (self.dilate(prediction) & truth).sum((1, 2))
            self.counts[domain] += torch.stack((
                true_positive, predicted - true_positive, labelled - true_positive,
                predicted_near_label, predicted, label_near_prediction, labelled,
                valid_map.sum().expand_as(true_positive),
            ), 1).double()

    def count_array(self, domain_ids=None) -> np.ndarray:
        """Counts summed over the given sources: array (thresholds, len(COUNT_NAMES))."""
        counts = self.counts if domain_ids is None else self.counts[list(domain_ids)]
        return counts.sum(0).cpu().numpy()

    def metrics(self, domain_ids=None) -> dict[str, np.ndarray]:
        return metrics_from_counts(self.count_array(domain_ids))


def metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Per-threshold metrics from an array (thresholds, len(COUNT_NAMES)) of pixel counts."""
    tp, fp, fn, near_label, predicted, near_prediction, labelled, pixels = counts.T
    epsilon = 1e-9
    tn = pixels - tp - fp - fn
    precision, recall = tp / (tp + fp + epsilon), tp / (tp + fn + epsilon)
    tol_precision, tol_recall = near_label / (predicted + epsilon), near_prediction / (labelled + epsilon)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + epsilon),
        "iou": tp / (tp + fp + fn + epsilon),
        "tol_precision": tol_precision, "tol_recall": tol_recall,
        "tol_f1": 2 * tol_precision * tol_recall / (tol_precision + tol_recall + epsilon),
        "accuracy": (tp + tn) / (pixels + epsilon),
        "false_positive_rate": fp / (fp + tn + epsilon),
    }


def basic_metrics(tp: float, fp: float, fn: float, pixels: float) -> dict[str, float]:
    """BASIC_METRICS from pixel counts at one threshold."""
    epsilon = 1e-9
    precision, recall = tp / (tp + fp + epsilon), tp / (tp + fn + epsilon)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + epsilon),
        "iou": tp / (tp + fp + fn + epsilon), "accuracy": (pixels - fp - fn) / (pixels + epsilon),
    }


def confusion_from_counts(counts_row: np.ndarray) -> dict[str, int]:
    """Pixel confusion matrix at one threshold: crack is the positive class."""
    tp, fp, fn = (int(round(float(value))) for value in counts_row[:3])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int(round(float(counts_row[7]))) - tp - fp - fn}


def at_threshold(metrics: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(metrics[name][index]) for name in METRIC_NAMES}


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Offsets of tiles of size `tile` covering `length`; the last one ends flush with the edge."""
    if length <= tile:
        return [0]
    return [*range(0, length - tile, stride), length - tile]


@torch.no_grad()
def predict_logits(model, images, gsd, use_amp: bool, tile: int = 0, overlap: int = 0, tile_batch: int = 1):
    """Float logits (N, 1, H, W) of the main output; `gsd` (N,) is the GSD of each image.

    Images with a side longer than `tile` are run as overlapping tiles, and the
    tile logits are blended with weights that fade toward each tile's border.
    That bounds memory on full-resolution photos, which do not fit whole, and
    matches tiled inference. `tile` 0 always runs the whole image. A tile is a
    crop of its image, so it has the image's GSD.
    """
    def run(batch, batch_gsd):
        with torch.autocast(device_type=batch.device.type, dtype=torch.float16, enabled=use_amp):
            return model(batch, batch_gsd).float()

    height, width = images.shape[-2:]
    if tile <= 0 or max(height, width) <= tile:
        return run(images, gsd)
    tile_h, tile_w, stride = min(tile, height), min(tile, width), tile - overlap
    windows = [(y, x) for y in tile_origins(height, tile_h, stride) for x in tile_origins(width, tile_w, stride)]

    def ramp(length):  # 1 in the tile's interior, falling linearly to its border
        position = torch.arange(length, device=images.device, dtype=torch.float32)
        return (torch.minimum(position, length - 1 - position) + 1).clamp(max=max(1, overlap)) / max(1, overlap)

    weight = ramp(tile_h)[:, None] * ramp(tile_w)[None, :]
    logits = torch.zeros((images.size(0), 1, height, width), device=images.device)
    weights = torch.zeros((height, width), device=images.device)
    for start in range(0, len(windows), tile_batch):
        chunk = windows[start:start + tile_batch]
        outputs = run(torch.cat([images[..., y:y + tile_h, x:x + tile_w] for y, x in chunk]),
                      gsd.repeat(len(chunk)))
        for (y, x), output in zip(chunk, outputs.split(images.size(0))):
            logits[..., y:y + tile_h, x:x + tile_w] += output * weight
            weights[y:y + tile_h, x:x + tile_w] += weight
    return logits / weights


@torch.no_grad()
def predict_probabilities(model, images, gsd, use_amp: bool, tta: bool, **tiling):
    """`tiling` holds the tile, overlap and tile_batch arguments of predict_logits."""
    logits = predict_logits(model, images, gsd, use_amp, **tiling)
    probabilities = torch.sigmoid(logits)
    if tta:  # a flip does not change the pixel size, so the GSD is the same for every view
        for dims in ((3,), (2,), (2, 3)):
            flipped = predict_logits(model, torch.flip(images, dims), gsd, use_amp, **tiling)
            probabilities += torch.sigmoid(torch.flip(flipped, dims))
        probabilities /= 4
    return logits, probabilities


@torch.no_grad()
def evaluate(model, loader, device, sweep: ThresholdSweep, use_amp: bool,
             loss_function=None, tta: bool = False, hook=None,
             gsd_transform=None, gsd_sweep: ThresholdSweep | None = None, gsd_group_ids=None, **tiling) -> float:
    """`gsd_transform` replaces the GSD given to the model (the GSD sanity check).
    `gsd_sweep` accumulates the same counts per GSD group, `gsd_group_ids[index]`, instead of per source."""
    model.eval()
    sweep.reset()
    if gsd_sweep is not None:
        gsd_sweep.reset()
    total_loss, samples = 0.0, 0
    for images, masks, valid, domain_ids, indices, gsd in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        masks, valid = masks.to(device, non_blocking=True), valid.to(device, non_blocking=True)
        gsd = gsd.to(device, non_blocking=True)
        if gsd_transform is not None:
            gsd = gsd_transform(gsd)
        logits, probabilities = predict_probabilities(model, images, gsd, use_amp, tta, **tiling)
        if loss_function is not None:
            total_loss += loss_function(logits, masks, valid).item() * images.size(0)
            samples += images.size(0)
        sweep.update(probabilities, masks, valid, domain_ids)
        if gsd_sweep is not None:
            gsd_sweep.update(probabilities, masks, valid, gsd_group_ids[indices])
        if hook is not None:
            hook(images, probabilities, masks, indices)
    return total_loss / max(1, samples)


# The GSD given to the model in each pass of the sanity check.
GSD_SANITY_VARIANTS = (
    ("true_gsd", "True GSD", None),
    ("gsd_x0.5", "GSD x 0.5", lambda gsd: gsd * 0.5),
    ("gsd_x2", "GSD x 2", lambda gsd: gsd * 2.0),
    ("unknown", "No GSD (null embedding)", lambda gsd: torch.full_like(gsd, UNKNOWN_GSD)),
)
GSD_SANITY_METRICS = ("f1", "tol_f1", "iou")


def gsd_sanity_check(model, loader, device, sweep: ThresholdSweep, use_amp: bool, tta: bool,
                     threshold_index: int, **tiling) -> dict:
    """Does the model use the GSD? Re-run the test set with a wrong GSD and with none.

    A model that has learned to use the GSD should lose accuracy when it is told half or double
    the true value. Tiles whose GSD is unknown get the null embedding in every pass. All passes
    use the threshold calibrated on validation with the true GSD.
    """
    variants = []
    for name, label, transform in GSD_SANITY_VARIANTS:
        evaluate(model, loader, device, sweep, use_amp, tta=tta, gsd_transform=transform, **tiling)
        metrics = at_threshold(sweep.metrics(), threshold_index)
        variants.append({"name": name, "label": label, **{key: metrics[key] for key in GSD_SANITY_METRICS}})
        print(f"GSD sanity check | {label:24s} F1 {metrics['f1']:.4f}  tolF1 {metrics['tol_f1']:.4f}  "
              f"IoU {metrics['iou']:.4f}")
    largest_change = max(abs(variant[key] - variants[0][key]) for variant in variants[1:]
                         for key in GSD_SANITY_METRICS)
    return {"threshold": float(sweep.thresholds[threshold_index]), "tta": tta, "variants": variants,
            "largest_change": largest_change, "ignores_gsd": largest_change < 0.005}


def prediction_writer(dataset, directory: Path, limit: int, threshold: float):
    """Save probability maps and overlays: yellow = hit, red = false positive, green = missed."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [0]

    def write(images, probabilities, masks, indices):
        for image, probability, mask, index in zip(images, probabilities, masks, indices.tolist()):
            if written[0] >= limit:
                return
            height, width = dataset.shapes[index]
            rgb = (image[:, :height, :width].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            probability = probability[0, :height, :width].cpu().numpy()
            truth, prediction = mask[0, :height, :width].cpu().numpy() > 0.5, probability >= threshold
            overlay = (rgb * 0.55).astype(np.uint8)
            overlay[truth & ~prediction] = (0, 220, 0)
            overlay[prediction & ~truth] = (230, 0, 0)
            overlay[prediction & truth] = (255, 220, 0)
            stem = dataset.pairs[index][0].stem
            cv2.imwrite(str(directory / f"{stem}_probability.png"), (probability * 255).astype(np.uint8))
            cv2.imwrite(str(directory / f"{stem}_overlay.png"),
                        cv2.cvtColor(np.hstack((rgb, overlay)), cv2.COLOR_RGB2BGR))
            written[0] += 1

    return write


# --------------------------------------------------------------------------- #
# Tables and figures: loss/metric curves, threshold sweep, PR, ROC, confusion matrix
# --------------------------------------------------------------------------- #
MODE_LABELS = {"no_tta": "no TTA", "tta": "flip TTA"}
PLOT_INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
# Fixed colour order (blue, orange, aqua): distinguishable with colour-vision deficiency.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
BLUE_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # one hue, light -> dark
PLOT_STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.edgecolor": PLOT_INK["axis"], "axes.labelcolor": PLOT_INK["secondary"],
    "axes.titlecolor": PLOT_INK["primary"], "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": PLOT_INK["grid"], "grid.linewidth": 0.8,
    "grid.linestyle": "-", "xtick.color": PLOT_INK["muted"], "ytick.color": PLOT_INK["muted"],
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False,
    "text.color": PLOT_INK["primary"],
}


def read_history(path: Path) -> dict[str, np.ndarray]:
    """training_history.csv as {column: values}; empty when the file is missing."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    history = {}
    for name in (rows[0] if rows else ()):
        try:
            history[name] = np.array([float(row[name]) for row in rows])
        except (TypeError, ValueError):
            continue
    return history


def save_evaluation_tables(folder: Path, thresholds, evaluation: dict) -> None:
    """The numbers behind the figures: evaluation_curves.csv and confusion_matrices.json."""
    folder.mkdir(parents=True, exist_ok=True)
    curve_names = (*METRIC_NAMES, "false_positive_rate")
    confusion = {}
    with (folder / "evaluation_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "split", "threshold", *COUNT_NAMES, "tn", *curve_names))
        for mode, data in evaluation.items():
            index = data["threshold_index"]
            confusion[mode] = {"threshold": float(thresholds[index])}
            for split, name in (("val", "validation"), ("test", "test")):
                counts, metrics = data[split], metrics_from_counts(data[split])
                for row, threshold in enumerate(thresholds):
                    matrix = confusion_from_counts(counts[row])
                    writer.writerow((mode, name, f"{threshold:.3f}", *(int(round(value)) for value in counts[row]),
                                     matrix["tn"], *(f"{metrics[key][row]:.6f}" for key in curve_names)))
                matrix = confusion_from_counts(counts[index])
                confusion[mode][name] = {
                    **matrix, **{key: float(metrics[key][index]) for key in curve_names},
                    "specificity": matrix["tn"] / max(1, matrix["tn"] + matrix["fp"]),
                }
    (folder / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2))


def save_plots(output: Path, thresholds, evaluation: dict, best_epoch: int, tolerance: int) -> list[str]:
    """Write every figure to <output>/plots and return the file names.

    `evaluation` maps "no_tta"/"tta" to {"threshold_index", "val", "test"} with
    count arrays from ThresholdSweep.count_array(). Epoch curves come from
    training_history.csv; figures needing columns an older history lacks are skipped.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    folder = output / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    ink, written = PLOT_INK, []
    blue, orange, aqua = SERIES_COLOURS
    thresholds = np.asarray(thresholds, dtype=float)

    def finish(figure, name, epoch_axis=False):
        if epoch_axis:  # epochs are whole numbers
            for axis in figure.axes:
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        figure.savefig(folder / name)
        plt.close(figure)
        written.append(name)

    def note(axis, text, offset=-0.22):
        axis.text(0, offset, text, transform=axis.transAxes, fontsize=8, color=ink["muted"], va="top")

    def mark(axis, x, text):
        axis.axvline(x, color=ink["muted"], linewidth=1.0, zorder=1)
        axis.annotate(text, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -3),
                      textcoords="offset points", va="top", fontsize=8.5, color=ink["secondary"])

    def lines(axis, x, series, end_values=True, pad=0.10):
        """series = [(label, values, colour)]. End values are written only where they do
        not collide; the legend always carries the identity."""
        for label, values, colour in series:
            axis.plot(x, values, color=colour, label=label)
        span = (x[-1] - x[0]) or 1.0
        axis.set_xlim(x[0] - 0.01 * span, x[-1] + (pad if end_values else 0.01) * span)
        if end_values:
            low, high = axis.get_ylim()
            placed = []
            for _, values, _ in sorted(series, key=lambda item: -item[1][-1]):
                if all(abs(values[-1] - other) > 0.06 * (high - low) for other in placed):
                    axis.annotate(f"{values[-1]:.3f}", (x[-1], values[-1]), xytext=(5, 0),
                                  textcoords="offset points", va="center", fontsize=8.5, color=ink["secondary"])
                    placed.append(values[-1])

    def legend_in_title_row(axis, columns):
        """Single panel: legend right of the left-aligned title, clear of the data."""
        axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=columns, borderaxespad=0.2,
                    handlelength=1.6, columnspacing=1.4)

    def legend_in_heading_row(axes, source_axis, columns):
        """Several panels: one legend at the right of the figure heading."""
        axes[-1].legend(*source_axis.get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(1.0, 1.09),
                        ncol=columns, borderaxespad=0, handlelength=1.6, columnspacing=1.4)

    def dot(axis, x, y, colour):
        axis.plot(x, y, "o", color=colour, markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=4)

    with plt.rc_context(PLOT_STYLE):
        history = read_history(output / "training_history.csv")
        if "epoch" in history and "train_loss" in history:
            epochs = history["epoch"]
            show_best = bool(best_epoch) and epochs[0] <= best_epoch <= epochs[-1]

            figure, axis = plt.subplots(figsize=(7, 4.2))
            lines(axis, epochs, [("Training", history["train_loss"], blue), ("Validation", history["val_loss"], orange)])
            if show_best:
                mark(axis, best_epoch, f"best epoch {best_epoch}")
            axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
            legend_in_title_row(axis, 2)
            note(axis, "Training loss also contains the auxiliary deep-supervision terms and is measured on augmented "
                       "crops,\nso it sits above the validation loss (full target images, main output only).")
            finish(figure, "loss_curves.png", epoch_axis=True)

            if all(f"{split}_{name}" in history for split in ("train", "val") for name in BASIC_METRICS):
                titles = {"precision": "Precision", "recall": "Recall", "f1": "F1 score", "iou": "IoU",
                          "accuracy": "Pixel accuracy"}
                figure, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
                for axis, name in zip(axes.flat, BASIC_METRICS):
                    lines(axis, epochs, [("Training", history[f"train_{name}"], blue),
                                         ("Validation", history[f"val_{name}"], orange)], pad=0.17)
                    if show_best:
                        mark(axis, best_epoch, "best")
                    axis.set_title(titles[name])
                for axis in (axes[1, 0], axes[1, 1], axes[0, 2]):
                    axis.set_xlabel("Epoch")
                axes[0, 2].tick_params(labelbottom=True)
                axes[1, 2].axis("off")
                axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
                                  title="Both at threshold 0.5")
                note(axes[1, 2], "Training: augmented crops, all sources.\nValidation: full target images, EMA weights.\n"
                                 "Pixel accuracy is dominated by background\nand stays near 1; judge cracks by F1 and IoU.",
                     offset=0.62)
                figure.suptitle("Training and validation metrics per epoch", x=0.07, ha="left",
                                fontsize=12, fontweight="semibold")
                finish(figure, "metric_curves_train_val.png", epoch_axis=True)

            if all(name in history for name in ("precision", "recall", "f1", "iou", "tol_f1")):
                figure, axes = plt.subplots(1, 3, figsize=(14, 4))
                for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                            (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                    lines(axis, epochs, [("Precision", history[prefix + "precision"], blue),
                                         ("Recall", history[prefix + "recall"], orange),
                                         ("F1", history[prefix + "f1"], aqua)], pad=0.14)
                    axis.set(xlabel="Epoch", title=title)
                # One series: the title names it, and a neutral ink keeps it from reading as "Precision".
                lines(axes[2], epochs, [("IoU", history["iou"], ink["secondary"])], pad=0.14)
                axes[2].set(xlabel="Epoch", title="IoU")
                legend_in_heading_row(axes, axes[0], 3)
                if show_best:
                    for axis in axes:
                        mark(axis, best_epoch, f"best epoch {best_epoch}")
                figure.suptitle("Validation metrics per epoch (target images, at each epoch's best threshold)",
                                x=0.07, ha="left", fontsize=12, fontweight="semibold")
                finish(figure, "validation_metric_curves.png", epoch_axis=True)

            if "lr" in history:
                figure, axis = plt.subplots(figsize=(7, 3.6))
                lines(axis, epochs, [("Learning rate", history["lr"], blue)], end_values=False)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
                axis.set(xlabel="Epoch", ylabel="Decoder learning rate", title="Learning-rate schedule (warm-up, cosine decay)")
                finish(figure, "learning_rate.png", epoch_axis=True)

        coverage_path = output / "contrast_coverage.json"
        if coverage_path.is_file():  # written by training in the "all" contrast mode
            coverage = json.loads(coverage_path.read_text())
            changes, draws = coverage["contrast_change_percent"], coverage["draws_per_level"]
            spacing = min(np.diff(changes)) if len(changes) > 1 else 1.0
            figure, axis = plt.subplots(figsize=(7.5, 3.8))
            axis.bar(changes, draws, width=0.6 * spacing, color=blue)
            axis.grid(axis="x", visible=False)
            ticks = changes[::2] if len(changes) > 12 else changes
            axis.set_xticks(ticks, [f"{value:+g}" if value else "0" for value in ticks])
            axis.set(xlabel="Contrast change (%)", ylabel="Training samples",
                     title=f"Training samples at each of the {coverage['levels']} contrast levels")
            note(axis, f"{coverage['images_trained_at_every_level']:,} of {coverage['training_images']:,} training "
                       f"images were trained at every level (mean {coverage['mean_draws_per_image']:.1f} draws per "
                       "image).", offset=-0.24)
            finish(figure, "contrast_level_coverage.png")

        colour_map = LinearSegmentedColormap.from_list("crack_blue", BLUE_RAMP)
        for mode, data in evaluation.items():
            label, index = MODE_LABELS.get(mode, mode), data["threshold_index"]
            selected = thresholds[index]
            splits = (("Validation", metrics_from_counts(data["val"]), orange),
                      ("Test", metrics_from_counts(data["test"]), aqua))
            validation = splits[0][1]

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
            for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                        (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                lines(axis, thresholds, [("Precision", validation[prefix + "precision"], blue),
                                         ("Recall", validation[prefix + "recall"], orange),
                                         ("F1", validation[prefix + "f1"], aqua)], end_values=False)
                mark(axis, selected, f"selected {selected:.3f}")
                axis.set(xlabel="Probability threshold", title=title, ylim=(0, 1))
            axes[0].set_ylabel("Score")
            legend_in_heading_row(axes, axes[0], 3)
            figure.suptitle(f"Validation threshold sweep ({label})", x=0.07, ha="left", fontsize=12,
                            fontweight="semibold")
            finish(figure, f"threshold_sweep_{mode}.png")

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
            for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                        (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                for name, metrics, colour in splits:
                    axis.plot(metrics[prefix + "recall"], metrics[prefix + "precision"], color=colour, label=name)
                    dot(axis, metrics[prefix + "recall"][index], metrics[prefix + "precision"][index], colour)
                axis.set(xlabel="Recall", title=title, xlim=(0, 1), ylim=(0, 1))
            axes[0].set_ylabel("Precision")
            legend_in_heading_row(axes, axes[0], 2)
            note(axes[0], f"Curves cover thresholds {thresholds[0]:.2f}-{thresholds[-1]:.2f}; "
                          f"dots mark the selected threshold {selected:.3f}.", offset=-0.2)
            figure.suptitle(f"Precision-recall curves ({label})", x=0.07, ha="left", fontsize=12,
                            fontweight="semibold")
            finish(figure, f"precision_recall_curve_{mode}.png")

            figure, axis = plt.subplots(figsize=(6.4, 4.4))
            for name, metrics, colour in splits:
                axis.plot(metrics["false_positive_rate"], metrics["recall"], color=colour, label=name)
                dot(axis, metrics["false_positive_rate"][index], metrics["recall"][index], colour)
            axis.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
                     title=f"ROC curves ({label})", xlim=(0, None), ylim=(0, 1))
            legend_in_title_row(axis, 2)
            background = 100 * (1 - data["test"][0, 6] / max(1.0, data["test"][0, 7]))
            note(axis, f"Thresholds {thresholds[0]:.2f}-{thresholds[-1]:.2f}; dots mark the selected threshold "
                       f"{selected:.3f}.\nBackground is {background:.1f} % of the test pixels, so the false "
                       "positive rate stays small.", offset=-0.19)
            finish(figure, f"roc_curve_{mode}.png")

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
            for axis, name, counts in ((axes[0], "Validation", data["val"]), (axes[1], "Test", data["test"])):
                cells = confusion_from_counts(counts[index])
                matrix = np.array([[cells["tp"], cells["fn"]], [cells["fp"], cells["tn"]]], dtype=float)
                share = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
                image = axis.imshow(share, cmap=colour_map, vmin=0, vmax=1)
                for row in range(2):
                    for column in range(2):
                        axis.text(column, row, f"{int(matrix[row, column]):,}\n{100 * share[row, column]:.2f} %",
                                  ha="center", va="center", fontsize=11,
                                  color="white" if share[row, column] > 0.5 else ink["primary"])
                axis.set_xticks([0, 1], ["Crack", "Background"])
                axis.set_yticks([0, 1], ["Crack", "Background"], rotation=90, va="center")
                axis.set(xlabel="Predicted", ylabel="Actual", title=f"{name} (threshold {selected:.3f})")
                axis.grid(False)
                axis.set_xticks([0.5], minor=True)
                axis.set_yticks([0.5], minor=True)
                axis.grid(which="minor", color="white", linewidth=2)  # surface gap between cells
                axis.tick_params(which="both", length=0, labelcolor=ink["secondary"])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
            bar.set_label("Share of the actual class (row)", color=ink["secondary"])
            bar.outline.set_visible(False)
            figure.suptitle(f"Pixel confusion matrices ({label}): count and share of each actual class",
                            x=0.07, ha="left", fontsize=12, fontweight="semibold")
            finish(figure, f"confusion_matrix_{mode}.png")
    return written


def save_gsd_sanity_plot(folder: Path, sanity: dict, tolerance: int) -> str:
    """Bar chart of the GSD sanity check, in the style of save_plots()."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    ink, variants = PLOT_INK, sanity["variants"]
    series = (("f1", "F1"), ("tol_f1", f"{tolerance}-px tolerance F1"), ("iou", "IoU"))
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(8.4, 4.4))
        width = 0.26
        for offset, ((key, label), colour) in enumerate(zip(series, SERIES_COLOURS)):
            positions = np.arange(len(variants)) + (offset - 1) * width
            values = [variant[key] for variant in variants]
            axis.bar(positions, values, width=0.88 * width, color=colour, label=label)  # gap between bars
            for position, value in zip(positions, values):
                axis.annotate(f"{value:.3f}", (position, value), xytext=(0, 2), textcoords="offset points",
                              ha="center", va="bottom", fontsize=8, color=ink["secondary"])
        axis.grid(axis="x", visible=False)
        axis.set_xticks(range(len(variants)), [variant["label"] for variant in variants])
        axis.tick_params(axis="x", length=0, labelcolor=ink["secondary"])
        highest = max(variant[key] for variant in variants for key, _ in series)
        axis.set(ylabel="Score", ylim=(0, min(1.0, 1.15 * highest) if highest > 0 else 1.0),
                 title="Test metrics by the GSD given to the model")
        axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, borderaxespad=0.2,
                    handlelength=1.6, columnspacing=1.4)
        verdict = ("The scores barely move: the model appears to ignore the GSD." if sanity["ignores_gsd"]
                   else "The scores change with the GSD: the model uses it.")
        axis.text(0, -0.14, f"Same weights and threshold ({sanity['threshold']:.3f}) in every pass; only the GSD "
                            f"input differs.\n{verdict} Largest change {sanity['largest_change']:.3f}.",
                  transform=axis.transAxes, fontsize=8, color=ink["muted"], va="top")
        figure.savefig(folder / "gsd_sanity_check.png")
        plt.close(figure)
    return "gsd_sanity_check.png"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def parameter_groups(model, learning_rate, encoder_factor, weight_decay):
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # No weight decay on BatchNorm weights and biases, nor on the FiLM generator: decay
            # pulls its outputs back to d_gamma = 0, beta = 0, that is, toward ignoring the GSD.
            # The generator is not part of the encoder, so it trains at the decoder rate.
            no_decay = parameter.ndim <= 1 or name.startswith("film.")
            groups[(name.startswith("encoder."), no_decay)].append(parameter)
    return [
        {"params": parameters, "lr": learning_rate * (encoder_factor if is_encoder else 1.0),
         "weight_decay": 0.0 if no_decay else weight_decay}
        for (is_encoder, no_decay), parameters in groups.items()
    ]


def warmup_cosine(total_steps: int, warmup_steps: int, minimum_ratio: float):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def json_safe(value):
    return json.loads(json.dumps(value, default=str))


def flag_name(name: str) -> str:
    """Attribute name of an operation's arguments: 'zoom-out' -> 'zoom_out' (+ '_p' / '_m')."""
    return name.replace("-", "_")


# Reference augmentation arguments. Their argparse default is None, meaning "not given": the
# preset decides, and for every preset but "none" that is the reference default.
REFERENCE_AUGMENTATION_ARGUMENTS = (
    "scale_p", "focus_p", "copy_paste_p", "synthetic_p", "fade_p", "affine_p", "elastic_p", "cutout_p",
    "max_rotation", "synthetic_label_width", "contrast_p", "contrast_min_change", "contrast_max_change",
    "contrast_mode", "contrast_step",
)


def resolve_augmentation(args) -> tuple[AugmentationOptions, dict]:
    """Preset first, then the individual flags on top of it.

    Fills the unset augmentation arguments of `args` with their resolved values and returns the
    options for the dataset and the full configuration that is stored with the run.
    """
    defaults, preset = AugmentationOptions(), args.aug_preset
    overrides = dict(NO_AUGMENTATION) if preset == "none" else {}
    if preset in PRESET_MAX_ROTATION:
        overrides["max_rotation"] = PRESET_MAX_ROTATION[preset]
    for name in REFERENCE_AUGMENTATION_ARGUMENTS:
        if getattr(args, name) is None:
            setattr(args, name, overrides.get(name, getattr(defaults, name)))
    probabilities, operations = preset_probabilities(preset), {}
    for name, operation in AUGMENTATIONS.items():
        probability = getattr(args, flag_name(name) + "_p")
        magnitude = getattr(args, flag_name(name) + "_m", None)
        probability = probabilities[name] if probability is None else probability
        magnitude = PRESET_MAGNITUDE[preset] if magnitude is None else magnitude
        if not (0 <= probability <= 1 and 0 <= magnitude <= 1):
            raise SystemExit(f"--{name}-p and --{name}-m must lie in [0, 1].")
        operations[name] = (float(probability), float(magnitude))
        setattr(args, flag_name(name) + "_p", float(probability))
        if operation.scaled:
            setattr(args, flag_name(name) + "_m", float(magnitude))
    randaugment = None
    if args.randaugment is not None:
        count, magnitude = args.randaugment
        if count != int(count) or not 0 <= count <= len(RANDAUGMENT_POOL) or not 0 <= magnitude <= 10:
            raise SystemExit(f"--randaugment N M needs a whole N in 0-{len(RANDAUGMENT_POOL)} and M in 0-10.")
        randaugment = (int(count), float(magnitude)) if count > 0 else None
    options = AugmentationOptions(
        args.scale_p, args.focus_p, args.copy_paste_p, args.synthetic_p, args.fade_p,
        args.affine_p, args.elastic_p, args.cutout_p, args.max_rotation, args.synthetic_label_width,
        args.contrast_p, args.contrast_min_change, args.contrast_max_change,
        args.contrast_mode, args.contrast_step,
        dihedral=preset != "none", photometric=preset != "none", ops=operations,
        label_safety=args.label_safety, randaugment=randaugment,
    )
    configuration = {
        "preset": preset,
        "order": ["scale", "crop", "mixing", "synthetic cracks and distractors", "geometric",
                  "orthomosaic artefacts", "photometric", "weather and lighting", "blur", "noise",
                  "compression", "occlusion"],
        "reference": {**{name: getattr(args, name) for name in REFERENCE_AUGMENTATION_ARGUMENTS},
                      "dihedral": options.dihedral, "photometric": options.photometric},
        "operations": {name: {"stage": AUGMENTATIONS[name].stage, "p": probability, "m": magnitude}
                       for name, (probability, magnitude) in operations.items()},
        "enabled_operations": [name for name, (probability, _) in operations.items() if probability > 0],
        "label_safety": {"enabled": bool(args.label_safety), "threshold_grey_levels": LABEL_SAFETY_GREY_LEVELS,
                         "cell_px": LABEL_SAFETY_CELL},
        "randaugment": None if randaugment is None else {"n": randaugment[0], "m": randaugment[1],
                                                         "pool": RANDAUGMENT_POOL},
    }
    return options, configuration


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET_PATH,
                        help=f"80/10/10 dataset root (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resume", action="store_true", help="Continue from <output>/last.pt")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="No training: evaluate <output>/best.pt on validation and test and write the "
                             "figures, curves and confusion matrices to <output>/plots (other files untouched)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the built-in checks (FiLM identity, augmentation alignment and GSD "
                             "bookkeeping, unknown GSD) on a temporary dummy dataset, then exit")
    # Model
    parser.add_argument("--encoder", choices=tuple(ENCODER_CHANNELS), default="resnet34")
    parser.add_argument("--no-pretrained", action="store_true", help="Random encoder initialisation")
    parser.add_argument("--no-line-prior", action="store_true",
                        help="Disable the black-hat/top-hat thin-line input channels")
    parser.add_argument("--no-deep-supervision", action="store_true")
    parser.add_argument("--no-film", action="store_true",
                        help="No GSD conditioning: the plain hairline U-Net, identical otherwise (ablation baseline)")
    parser.add_argument("--film-skips", action="store_true",
                        help="Also modulate the five skip connections, not only the five decoder block outputs")
    parser.add_argument("--gsd-dropout", type=float, default=0.15,
                        help="Training: probability of replacing a known GSD by the null embedding")
    # GSD data
    parser.add_argument("--gsd-csv", type=Path, default=None,
                        help="CSV with the columns pattern,gsd_mm_per_px. pattern is a regular expression "
                             "searched in the image file name (e.g. ^EH104_); the first matching row gives "
                             "the GSD, no match means unknown. Without it every tile is unknown")
    # Optimisation
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256,
                        help="Training crop size at native resolution; also the inference tile size")
    # CNN settings (those of train_unet_hairline.py), not the transformer fine-tuning ones.
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw",
                        help="adamw: decoupled weight decay (default, as the reference). adam: plain Adam, "
                             "where --weight-decay acts as L2 regularisation")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else min(4, os.cpu_count() or 1))
    # Loss
    parser.add_argument("--loss", choices=("hairline", "focal"), default="hairline",
                        help="hairline: BCE + Tversky + clDice (default, as the reference). focal: binary "
                             "focal loss + the same clDice term")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="--loss focal: weight of the crack class")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="--loss focal: focusing exponent")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--tversky-weight", type=float, default=1.0)
    parser.add_argument("--cldice-weight", type=float, default=0.5)
    parser.add_argument("--cldice-warmup-epochs", type=int, default=5,
                        help="Ramp clDice in linearly; it is noisy while predictions are still blobs")
    parser.add_argument("--false-positive-weight", type=float, default=0.4)
    parser.add_argument("--false-negative-weight", type=float, default=0.6,
                        help="> false-positive weight favours recall of faint cracks")
    parser.add_argument("--skeleton-iterations", type=int, default=10)
    parser.add_argument("--aux-weights", type=float, nargs=2, default=(0.4, 0.2))
    # Evaluation
    parser.add_argument("--tolerance", type=int, default=2, help="Pixel tolerance for tol_* metrics")
    parser.add_argument("--select-metric", choices=METRIC_NAMES, default="tol_f1",
                        help="Metric for checkpoint selection and threshold calibration")
    parser.add_argument("--select-pattern", default="",
                        help="Regex on validation file names defining the target domain (model selection "
                             r"and threshold), e.g. '^EH\d' for the EH104 tiles only. Default: empty = "
                             "all validation data, every source in the dataset")
    parser.add_argument("--no-tta", action="store_true", help="Disable flip TTA for calibration/test")
    parser.add_argument("--eval-batch-megapixels", type=float, default=2.0,
                        help="An evaluation batch holds at most this many megapixels "
                             "(and at most --eval-batch-size images)")
    parser.add_argument("--eval-tile", type=int, default=1024,
                        help="Validation/test images with a side above this are evaluated as overlapping "
                             "tiles of this size, because full-resolution photos do not fit in memory "
                             "whole. Smaller images still run whole; 0 disables tiling")
    parser.add_argument("--eval-tile-overlap", type=int, default=128, help="Overlap of those tiles in pixels")
    parser.add_argument("--save-predictions", type=int, default=40, help="Test overlays to save")
    parser.add_argument("--gsd-sanity-check", action="store_true",
                        help="After the test pass, re-run the test set with the true GSD, GSD x 0.5, GSD x 2 "
                             "and no GSD, and write gsd_sanity_check.json and plots/gsd_sanity_check.png")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug: subsample train")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Debug: subsample val/test")
    # Augmentation: the reference's arguments. No default here means "as the preset says", which is
    # the reference default shown in the help for every preset except "none".
    defaults = AugmentationOptions()
    parser.add_argument("--aug-preset", choices=PRESET_NAMES, default="reference",
                        help="reference: exactly the reference augmentation (default). none: random crop "
                             "only. light / medium / heavy: the reference plus the extended library (table "
                             "above). all: every operation at low probability, a stress test")
    parser.add_argument("--target-weight", type=float, default=1.5,
                        help="Sampling weight of target-domain training images relative to others")
    parser.add_argument("--scale-p", type=float, help=f"(reference default {defaults.scale_p})")
    parser.add_argument("--focus-p", type=float, help=f"(reference default {defaults.focus_p})")
    parser.add_argument("--copy-paste-p", type=float, help=f"(reference default {defaults.copy_paste_p})")
    parser.add_argument("--synthetic-p", type=float, help=f"(reference default {defaults.synthetic_p})")
    parser.add_argument("--fade-p", type=float, help=f"(reference default {defaults.fade_p})")
    parser.add_argument("--affine-p", type=float, help=f"(reference default {defaults.affine_p})")
    parser.add_argument("--elastic-p", type=float, help=f"(reference default {defaults.elastic_p})")
    parser.add_argument("--cutout-p", type=float, help=f"(reference default {defaults.cutout_p})")
    parser.add_argument("--max-rotation", type=float,
                        help=f"Largest rotation of the affine in degrees, up to 180 for a full rotation with "
                             f"reflect padding (reference default {defaults.max_rotation:g})")
    parser.add_argument("--synthetic-label-width", type=int,
                        help=f"(reference default {defaults.synthetic_label_width})")
    parser.add_argument("--contrast-mode", choices=("all", "random", "off"),
                        help="all: every training image is cycled through every contrast level from "
                             "-max to +max in --contrast-step steps, including the original (reference "
                             "default, 21 levels). random: a random change on a share --contrast-p of the "
                             "images. off: no contrast change (the 'none' preset)")
    parser.add_argument("--contrast-step", type=float,
                        help="Spacing of the contrast levels in 'all' mode (default 0.01 = 1 %%)")
    parser.add_argument("--contrast-p", type=float,
                        help="'random' mode only: probability of a contrast change per training image")
    parser.add_argument("--contrast-min-change", type=float,
                        help="Smallest contrast change, as a fraction (default 0.01 = 1 %%)")
    parser.add_argument("--contrast-max-change", type=float,
                        help="Largest contrast change, as a fraction (default 0.10 = 10 %%); "
                             "each change is an increase or a decrease with equal probability")
    # Augmentation: the extended library, one probability (and magnitude) per registered operation.
    library = parser.add_argument_group(
        "extended augmentation library",
        "--<name>-p: probability per training image (default: the preset's, 0 in 'reference'). "
        "--<name>-m: magnitude in [0, 1] scaling the range given below (default: the preset's).")
    for name, operation in AUGMENTATIONS.items():
        note = " [OFF in light/medium/heavy: harms crack labels]" if operation.default_off else ""
        library.add_argument(f"--{name}-p", type=float, metavar="P", help=f"{operation.stage}: {operation.help}{note}")
        if operation.scaled:
            library.add_argument(f"--{name}-m", type=float, metavar="M", help=argparse.SUPPRESS)
    parser.add_argument("--label-safety", action=argparse.BooleanOptionalAction, default=True,
                        help="After a contrast-reducing operation of the extended library, exclude cracks "
                             "that are no longer visible (below 4 grey levels of local contrast) from the loss")
    parser.add_argument("--randaugment", type=float, nargs=2, metavar=("N", "M"),
                        help="RandAugment: N random photometric / weather / blur / noise / compression "
                             "operations per image at magnitude M (0-10). Default: off")
    parser.add_argument("--preview-augmentations", type=int, default=0, metavar="K",
                        help="Before training, save one panel per enabled augmentation for K training tiles "
                             "to <output>/augmentation_preview")
    parser.add_argument("--preview-only", action="store_true", help="Exit after --preview-augmentations")
    # Jupyter injects ``-f <kernel.json>``. Because ``dataset`` is an optional
    # positional argument, remove the complete kernel pair before parsing.
    raw_arguments, filtered_arguments, index = sys.argv[1:], [], 0
    while index < len(raw_arguments):
        if raw_arguments[index] == "-f":
            index += 2
        else:
            filtered_arguments.append(raw_arguments[index])
            index += 1
    args = parser.parse_args(filtered_arguments)
    if args.image_size % 32:
        parser.error("--image-size must be divisible by 32 for a ResNet encoder.")
    if args.eval_tile % 32 or not 0 <= args.eval_tile_overlap < max(1, args.eval_tile):
        parser.error("--eval-tile must be divisible by 32 and larger than --eval-tile-overlap.")
    if args.eval_batch_megapixels <= 0:
        parser.error("--eval-batch-megapixels must be greater than zero.")
    if not 0 <= args.gsd_dropout <= 1:
        parser.error("--gsd-dropout must lie in [0, 1].")
    if args.preview_only and args.preview_augmentations <= 0:
        parser.error("--preview-only needs --preview-augmentations K.")
    return args


def check_augmentation_arguments(args) -> None:
    """The reference's checks, on the resolved values."""
    if not 0 <= args.contrast_min_change <= args.contrast_max_change < 1:
        raise SystemExit("Contrast changes must satisfy 0 <= --contrast-min-change <= --contrast-max-change < 1.")
    if args.contrast_step <= 0:
        raise SystemExit("--contrast-step must be greater than zero.")
    if not 0 <= args.max_rotation <= 180:
        raise SystemExit("--max-rotation must lie in [0, 180] degrees.")


# --------------------------------------------------------------------------- #
# Self-tests (--self-test)
# --------------------------------------------------------------------------- #
# In the test tiles the image encodes its own label: background 100, crack 220. A resampled
# image value above 100 + 0.4 * 120 is then exactly a resampled mask value above 0.4, so image
# and mask can be compared after any geometric operation.
SELF_TEST_BACKGROUND, SELF_TEST_CRACK = 100, 220
SELF_TEST_CUT = SELF_TEST_BACKGROUND + 0.4 * (SELF_TEST_CRACK - SELF_TEST_BACKGROUND)
# The operations that change the pixel size, and with it the GSD. Every other one must leave it.
SCALE_CHANGING = {"rescale", "affine", "resized-crop", "zoom-out", "perspective", "mosaic"}
GEOMETRIC_STAGES = ("mixing", "rescaling", "warp")


def encoded_tile(mask: np.ndarray) -> Tile:
    image = np.full((*mask.shape, 3), SELF_TEST_BACKGROUND, np.uint8)
    image[mask > 0] = SELF_TEST_CRACK
    return Tile(image, mask, np.ones_like(mask))


def write_self_test_dataset(root: Path, generator: random.Random) -> None:
    """Three sources: EH104 and EH205 (one source family, two GSDs) and F12 (no GSD)."""
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        for folder in ("images", "masks"):
            (root / split / folder).mkdir(parents=True)
        for prefix in ("EH104", "EH205", "F12"):
            for number in range(count):
                mask = np.zeros((280, 300), np.uint8)  # not a multiple of 32: evaluation must pad
                for _ in range(3):
                    points = [(generator.randint(0, 299), generator.randint(0, 279)) for _ in range(3)]
                    cv2.polylines(mask, [np.int32(points)], False, 1, 5)
                name = f"{prefix}_r{number:04d}_c{generator.randint(0, 99):04d}.png"
                tile = encoded_tile(mask)
                cv2.imwrite(str(root / split / "images" / name), tile.image)
                cv2.imwrite(str(root / split / "masks" / name), mask * 255)


def run_self_tests() -> None:
    """Checks of the FiLM model, of every augmentation's labels and GSD bookkeeping, and of the
    unknown-GSD path. Raises SystemExit with the failed checks, if any."""
    import tempfile

    seed_everything(0)
    failures = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}" + (f" ({detail})" if detail and not passed else ""))
        if not passed:
            failures.append(name)

    def binary(array) -> bool:
        return array.dtype == np.uint8 and set(np.unique(array).tolist()) <= {0, 1}

    def well_formed(tile: Tile) -> bool:
        return (tile.image.dtype == np.uint8 and tile.image.ndim == 3 and tile.image.shape[2] == 3
                and tile.image.shape[:2] == tile.mask.shape == tile.valid.shape
                and binary(tile.mask) and binary(tile.valid))

    def aligned(tile: Tile) -> bool:
        """The label read back from the image equals the mask wherever the tile is supervised."""
        wrong = ((tile.image[..., 0] >= SELF_TEST_CUT) != (tile.mask > 0)) & (tile.valid > 0)
        return int(wrong.sum()) <= 0.04 * int(tile.mask.sum()) + 8

    print("GSD table")
    table = GSDTable([("^EH104_", 0.5), ("^EH", 0.8)])
    check("first matching pattern gives the GSD", table.lookup("EH104_r0001_c0002.png") == 0.5
          and table.lookup("EH205_r0001_c0002.png") == 0.8)
    check("no match is unknown (NaN)", math.isnan(table.lookup("F12_r0001_c0002.png")))

    print("Preset table of the module docstring")
    documented = {}
    for line in __doc__.splitlines():
        cells = line.split()
        if len(cells) == 6 and cells[0].rstrip("*") in AUGMENTATIONS and cells[1] in STAGES:
            documented[cells[0].rstrip("*")] = (cells[0].endswith("*"), cells[1], *map(float, cells[2:]))
    coded = {name: (operation.default_off, operation.stage,
                    *(preset_probabilities(preset)[name] for preset in ("light", "medium", "heavy", "all")))
             for name, operation in AUGMENTATIONS.items()}
    check("lists every operation with the probabilities used", documented == coded,
          str({name for name in coded if documented.get(name) != coded[name]}))

    print("FiLM model")
    kwargs = dict(encoder_name="resnet18", pretrained=False, gsd_log_mean=math.log(0.6), gsd_log_std=0.4)
    for skips in (False, True):
        film_model = HairlineFiLMUNet(use_film=True, film_skips=skips, **kwargs)
        plain_model = HairlineFiLMUNet(use_film=False, **kwargs)
        missing = plain_model.load_state_dict(film_model.state_dict(), strict=False)
        images = torch.rand(3, 3, 96, 128)
        gsd = torch.tensor([0.5, float("nan"), 1.7])
        label = "decoder + skips" if skips else "decoder"
        check(f"only the FiLM generator separates the two models ({label})",
              not missing.missing_keys and all(key.startswith("film.") for key in missing.unexpected_keys))
        film_model.eval(), plain_model.eval()
        with torch.no_grad():
            check(f"zero-initialised FiLM equals --no-film, eval ({label})",
                  torch.equal(film_model(images, gsd), plain_model(images, gsd)))
        film_model.train(), plain_model.train()
        with torch.no_grad():
            torch.manual_seed(1)
            film_logits, film_auxiliary = film_model(images, gsd)
            torch.manual_seed(1)
            plain_logits, plain_auxiliary = plain_model(images, gsd)
        check(f"zero-initialised FiLM equals --no-film, train with deep supervision ({label})",
              torch.equal(film_logits, plain_logits)
              and all(torch.equal(a, b) for a, b in zip(film_auxiliary, plain_auxiliary)))
    # Give the generator non-zero heads, as training would, and look at what it reacts to.
    film_model.eval()
    with torch.no_grad():
        for head in film_model.film.heads.values():
            head.weight.normal_(0, 0.1)
        images = torch.rand(2, 3, 96, 128)
        nan = torch.full((2,), float("nan"))
        unknown, known = film_model(images, nan), film_model(images, torch.tensor([0.5, 0.5]))
        other = film_model(images, torch.tensor([1.5, 1.5]))
        check("a trained FiLM model reacts to the GSD", not torch.allclose(known, other))
        check("unknown GSD (NaN, None) runs and gives finite logits",
              bool(torch.isfinite(unknown).all()) and torch.equal(unknown, film_model(images)))
        film_model.film.null_embedding.add_(1.0)
        check("unknown GSD uses the null embedding", not torch.allclose(unknown, film_model(images, nan)))
        check("known GSD does not use the null embedding",
              torch.equal(known, film_model(images, torch.tensor([0.5, 0.5]))))
        mixed = film_model(images, torch.tensor([0.5, float("nan")]))
        check("known and unknown GSD mix in one batch",
              torch.allclose(mixed[0], known[0], atol=1e-4) and not torch.allclose(mixed[1], known[1]))
        film_model.train()
        film_model.gsd_dropout = 1.0
        dropped = film_model(images, torch.tensor([0.5, 0.5]))[0]
        film_model.gsd_dropout = 0.0
        kept = film_model(images, torch.tensor([0.5, 0.5]))[0]
        null = film_model(images, nan)[0]
        check("GSD dropout replaces a known GSD by the null embedding",
              torch.allclose(dropped, null, atol=1e-4) and not torch.allclose(kept, null))

    print("Augmentations on a label-encoded tile")
    mask = np.zeros((256, 256), np.uint8)
    cv2.circle(mask, (128, 128), 22, 1, -1)
    cv2.line(mask, (94, 158), (162, 162), 1, 6)
    base = encoded_tile(mask)
    partner_mask = np.zeros((256, 256), np.uint8)
    cv2.line(partner_mask, (20, 30), (230, 200), 1, 5)
    partner = encoded_tile(partner_mask)
    base.partner = lambda: (partner.image, partner.mask, partner.valid)

    def apply_reference(name: str, tile: Tile) -> None:
        if name == "rescale":
            tile.image, tile.mask, tile.valid, factor = rescale(tile.image, tile.mask, tile.valid,
                                                                random.uniform(0.75, 1.25))
        elif name == "crop":
            tile.image, tile.mask, tile.valid = random_crop(tile.image, tile.mask, tile.valid, 192, 0.6)
        elif name == "dihedral":
            tile.image, tile.mask, tile.valid = dihedral(tile.image, tile.mask, tile.valid)
        elif name == "affine":
            tile.image, tile.mask, tile.valid, factor = random_affine(tile.image, tile.mask, tile.valid, 20.0)
        elif name == "elastic":
            tile.image, tile.mask, tile.valid = elastic(tile.image, tile.mask, tile.valid)
        if name in ("rescale", "affine"):
            tile.scale *= factor

    geometric = ["rescale", "crop", "dihedral", "affine", "elastic", *stage_names(*GEOMETRIC_STAGES), "seam"]
    for name in geometric:
        problems = []
        for trial in range(6):
            tile = base.copy()
            if name in AUGMENTATIONS:
                AUGMENTATIONS[name].function(tile, 1.0)
            else:
                apply_reference(name, tile)
            if not well_formed(tile):
                problems.append("shape or non-binary labels")
            if name != "mixup" and not aligned(tile):  # a blend has no single label to read back
                problems.append("image and mask are not aligned")
            if name in SCALE_CHANGING:
                if name == "mosaic":
                    expected = 0.5
                else:  # the crack area grows with the square of the scale
                    expected = math.sqrt(tile.mask.sum() / mask.sum())
                if abs(tile.scale / expected - 1) > 0.08:
                    problems.append(f"scale {tile.scale:.3f}, measured {expected:.3f}")
                if not math.isclose(0.5 / tile.scale, 0.5 / expected, rel_tol=0.08) or not math.isnan(
                        UNKNOWN_GSD / tile.scale):
                    problems.append("GSD not updated")
            elif tile.scale != 1.0:
                problems.append(f"scale {tile.scale} from an operation that keeps the pixel size")
        note = "GSD / scale" if name in SCALE_CHANGING else "GSD unchanged"
        check(f"{name}: aligned, binary, {note}", not problems, "; ".join(sorted(set(problems))))
    zoomed = base.copy()
    AUGMENTATIONS["zoom-out"].function(zoomed, 1.0)
    check("zoom-out: the padding is not supervised",
          abs(zoomed.valid.mean() - zoomed.scale ** 2) < 0.03, f"{zoomed.valid.mean():.3f} vs {zoomed.scale ** 2:.3f}")
    rotated = [base.copy() for _ in range(6)]
    for tile in rotated:
        apply_reference("affine", tile)
    check("affine: warped-in pixels are not supervised",
          any(tile.valid.min() == 0 for tile in rotated) and all(tile.valid.max() == 1 for tile in rotated))

    print("Every other operation, at magnitudes 0.3 and 1")
    textured = base.copy()
    textured.image = to_uint8(textured.image.astype(np.float32) + np.random.normal(0, 6, textured.image.shape))
    textured.partner = base.partner
    keep_mask = [name for name, operation in AUGMENTATIONS.items()
                 if operation.stage not in (*GEOMETRIC_STAGES, "cracks") and name not in ("seam", "nodata-border")]
    for name, operation in AUGMENTATIONS.items():
        problems = []
        for magnitude in (0.3, 1.0):
            tile = textured.copy()
            operation.function(tile, magnitude)
            if not well_formed(tile):
                problems.append("shape or non-binary labels")
            if name in keep_mask and not np.array_equal(tile.mask, textured.mask):
                problems.append("mask changed")
            if name not in SCALE_CHANGING and tile.scale != 1.0:
                problems.append("scale changed")
            if operation.stage == "occlusion" and tile.valid.all():
                problems.append("nothing hidden")
            if operation.stage == "cracks" and not (tile.mask >= textured.mask).all():
                problems.append("labels removed")
        check(f"{name} ({operation.stage})", not problems, "; ".join(sorted(set(problems))))
    border = textured.copy()
    AUGMENTATIONS["nodata-border"].function(border, 1.0)
    black = border.image.max(2) == 0
    check("nodata-border: black, background, supervised",
          black.any() and not border.mask[black].any() and border.valid[black].all())
    painted = textured.copy()
    painted.mask[:] = 1
    AUGMENTATIONS["painted-line"].function(painted, 1.0)
    check("painted-line: the cracks it covers leave the loss", 0 < painted.valid.sum() < painted.valid.size)

    print("Label safety")
    faint_mask = np.zeros((128, 128), np.uint8)
    cv2.line(faint_mask, (10, 20), (118, 100), 1, 3)
    crack = np.zeros((128, 128), np.uint8)
    cv2.line(crack, (10, 20), (118, 100), 1, 1)
    faint = np.full((128, 128, 3), 130, np.uint8)
    faint[crack > 0] = 70
    ones = np.ones_like(faint_mask)
    _, excluded, checked = label_safety_check(faint, faint, faint_mask, ones)
    check("an unchanged image loses no label", excluded == 0 and checked == faint_mask.sum())
    erased = cv2.GaussianBlur(faint, (0, 0), 8)
    valid, excluded, checked = label_safety_check(faint, erased, faint_mask, ones)
    check("a crack blurred away leaves the loss", excluded > 0.9 * checked, f"{excluded} of {checked}")
    check("with a 2-px band around it", (valid == 0).sum() > faint_mask.sum() and binary(valid))
    _, excluded, _ = label_safety_check(erased, erased, faint_mask, ones)
    check("a crack that was never visible keeps its label", excluded == 0)

    print("Dataset, augmentation and model")
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        write_self_test_dataset(root, random.Random(0))
        pairs = collect_pairs(root / "train")
        families = [source_family(image) for image, _ in pairs]
        domain_index = {family: index for index, family in enumerate(sorted(set(families)))}
        table = GSDTable([("^EH104_", 0.5), ("^EH205_", 0.8)])
        gsds = [table.lookup(image.name) for image, _ in pairs]
        domains = [domain_index[family] for family in families]

        def dataset(training=True, **changes) -> CrackSegmentationDataset:
            settings = dict(scale_p=0.0, copy_paste_p=0.0, synthetic_p=0.0, fade_p=0.0, affine_p=0.0,
                            elastic_p=0.0, cutout_p=0.0, contrast_mode="off", dihedral=False, photometric=False)
            settings.update(changes)
            return CrackSegmentationDataset(pairs, domains, gsds, training=training, crop_size=256,
                                            options=AugmentationOptions(**settings))

        def known(prefix: str) -> list[int]:
            return [index for index, (image, _) in enumerate(pairs) if image.name.startswith(prefix)]

        def sample_tile(sample) -> Tile:
            image = (sample[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            return Tile(image, sample[1][0].numpy().astype(np.uint8), sample[2][0].numpy().astype(np.uint8))

        plain = dataset()
        check("EH104 and EH205 are one source family", len(set(families)) == 2, str(sorted(set(families))))
        check("mixing groups hold one source family and one GSD",
              len(plain.members) == 3 and all(len({(families[index], gsd_key(gsds[index])) for index in members}) == 1
                                              for members in plain.members.values()))
        check("partners come from the tile's own group",
              all(set(plain.members[plain.groups[index]]) == set(known(pairs[index][0].name.split("_")[0]))
                  for index in range(len(pairs))))
        samples = [plain[index] for index in range(len(pairs))]
        check("the dataset returns (image, mask, valid, domain, index, gsd)",
              all(len(sample) == 6 and sample[0].shape == (3, 256, 256) for sample in samples))
        check("no augmentation: the GSD is the table's",
              all(float(samples[index][5]) == 0.5 for index in known("EH104"))
              and all(math.isclose(float(samples[index][5]), 0.8, rel_tol=1e-6) for index in known("EH205")))
        check("no augmentation: tiles without a GSD are NaN",
              all(math.isnan(float(samples[index][5])) for index in known("F12")))
        check("no augmentation: image and label stay aligned", all(aligned(sample_tile(sample)) for sample in samples))

        scaled = dataset(scale_p=1.0)
        values = [float(scaled[index][5]) for index in known("EH104") for _ in range(4)]
        check("rescale: GSD / scale", all(0.5 / 1.26 <= value <= 0.5 / 0.74 and value != 0.5 for value in values))
        mosaics = dataset(ops={"mosaic": (1.0, 1.0)})
        samples = [mosaics[index] for index in range(len(pairs))]
        check("mosaic: GSD x 2", all(math.isclose(float(samples[index][5]), 1.6, rel_tol=1e-6)
                                     for index in known("EH205")))
        check("mosaic: unknown stays unknown", all(math.isnan(float(samples[index][5])) for index in known("F12")))
        check("mosaic: four aligned tiles", all(aligned(sample_tile(sample)) for sample in samples))
        cutmixes = dataset(ops={"cutmix": (1.0, 1.0)})
        samples = [cutmixes[index] for index in range(len(pairs))]
        check("cutmix: aligned, hard labels, GSD unchanged",
              all(aligned(sample_tile(sample)) and set(sample[1].unique().tolist()) <= {0.0, 1.0}
                  for sample in samples) and all(float(samples[index][5]) == 0.5 for index in known("EH104")))
        pasted = dataset(scale_p=1.0, copy_paste_p=1.0)
        check("copy-paste on a rescaled tile runs", all(len(pasted[index]) == 6 for index in range(len(pairs))))

        everything = {name: (0.3, 1.0) for name in AUGMENTATIONS}
        stress = CrackSegmentationDataset(pairs, domains, gsds, training=True, crop_size=256,
                                          options=AugmentationOptions(ops=everything, randaugment=(2, 9.0)))
        problems = []
        for _ in range(4):
            for index in range(len(pairs)):
                image, mask, valid, _, _, gsd = stress[index]
                if image.shape != (3, 256, 256) or not bool(torch.isfinite(image).all()):
                    problems.append("image")
                if not (set(mask.unique().tolist()) <= {0.0, 1.0} and set(valid.unique().tolist()) <= {0.0, 1.0}
                        and bool((mask <= valid).all())):
                    problems.append("labels")
                if math.isnan(float(gsd)) != (index in known("F12")) or (not math.isnan(float(gsd)) and gsd <= 0):
                    problems.append("gsd")
        check("every operation at once: valid tiles, NaN GSD stays NaN", not problems, str(sorted(set(problems))))
        counts = stress.augmentation_counts()
        check("augmentation counts are recorded",
              counts["training_samples_drawn"] == 4 * len(pairs) and sum(counts["fired"].values()) > 0
              and counts["label_safety"]["checks"] > 0)

        loader = DataLoader(stress, batch_size=6, shuffle=True, drop_last=True)
        images, masks, valid, _, _, gsd = next(iter(loader))
        model = HairlineFiLMUNet(**kwargs)
        model.train()
        for name, loss_function in (("hairline", HairlineCrackLoss()), ("focal", FocalCrackLoss())):
            model.zero_grad()
            loss = loss_function(model(images, gsd), masks, valid)
            loss.backward()
            check(f"{name} loss on a batch with known and unknown GSD is finite, gradients reach FiLM",
                  bool(torch.isfinite(loss)) and model.film.embed.weight.grad is not None
                  and bool(torch.isfinite(model.film.null_embedding.grad).all()))
        groups = parameter_groups(model, 3e-4, 0.1, 1e-4)
        film_parameters = {id(parameter) for parameter in model.film.parameters()}
        check("FiLM generator: decoder learning rate, no weight decay",
              all(group["lr"] == 3e-4 and group["weight_decay"] == 0.0 for group in groups
                  if film_parameters & {id(parameter) for parameter in group["params"]}))

        evaluation = CrackSegmentationDataset(collect_pairs(root / "test"), [0] * 6, [0.5, 0.8, UNKNOWN_GSD] * 2)
        image, mask, valid, _, _, gsd = evaluation[0]
        check("evaluation: full image padded to a multiple of 32, native GSD",
              image.shape == (3, 288, 320) and float(gsd) == 0.5 and float(valid.sum()) == 280 * 300)
        model.eval()
        batch, batch_gsd = torch.stack([evaluation[index][0] for index in range(3)]), torch.tensor(
            [0.5, 0.8, UNKNOWN_GSD])
        whole = predict_logits(model, batch, batch_gsd, False)
        tiled = predict_logits(model, batch, batch_gsd, False, tile=160, overlap=64, tile_batch=3)
        check("tiled evaluation passes each image's GSD to its tiles",
              whole.shape == tiled.shape == (3, 1, 288, 320) and bool(torch.isfinite(tiled).all()))

        shown = save_augmentation_preview(stress, root / "preview", 1, 0)
        check("preview: one panel per enabled augmentation and an overview",
              len(list((root / "preview").glob("*.png"))) == len(shown) + 1 and len(shown) == len(PIPELINE_ORDER))

    if failures:
        raise SystemExit(f"\nSelf-test FAILED ({len(failures)}): " + "; ".join(failures))
    print("\nAll self-tests passed.")


def main():
    args = parse_arguments()
    if args.self_test:
        run_self_tests()
        return
    options, augmentation = resolve_augmentation(args)
    check_augmentation_arguments(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = args.output / "last.pt", args.output / "best.pt"
    stored = None
    if args.evaluate_only:
        if not best_path.is_file():
            raise SystemExit(f"--evaluate-only needs a trained model, but {best_path} does not exist.")
        stored = torch.load(best_path, map_location="cpu", weights_only=False)

    # ---------------- GSD table ---------------- #
    if args.gsd_csv is not None:
        gsd_table = GSDTable.from_csv(args.gsd_csv)
        print(f"GSD table: {len(gsd_table.rows)} pattern(s) from {args.gsd_csv}.")
    elif stored is not None and stored.get("gsd_csv"):
        gsd_table = GSDTable(stored["gsd_csv"])  # evaluate with the GSD values the model was trained with
        print(f"GSD table: {len(gsd_table.rows)} pattern(s) stored in {best_path.name}.")
    else:
        gsd_table = GSDTable()
    if not args.evaluate_only:
        (args.output / "configuration.json").write_text(json.dumps(json_safe(
            {**vars(args), "augmentation": augmentation, "gsd_csv_rows": gsd_table.rows}), indent=2))

    splits = {name: collect_pairs(args.dataset / name) for name in ("train", "val", "test")}
    subsample = random.Random(args.seed)
    if args.max_train_samples:
        splits["train"] = subsample.sample(splits["train"], min(args.max_train_samples, len(splits["train"])))
    if args.max_eval_samples:
        for name in ("val", "test"):
            splits[name] = subsample.sample(splits[name], min(args.max_eval_samples, len(splits[name])))
    families = {name: [source_family(image) for image, _ in pairs] for name, pairs in splits.items()}
    domains = sorted(set().union(*families.values()))
    domain_index = {domain: index for index, domain in enumerate(domains)}
    # The GSD of a tile is that of the first CSV pattern found in its image file name; NaN without one.
    gsds = {name: [gsd_table.lookup(image.name) for image, _ in pairs] for name, pairs in splits.items()}

    pattern = re.compile(args.select_pattern) if args.select_pattern else None
    target_domains = sorted({
        family for family, (image, _) in zip(families["val"], splits["val"])
        if pattern is None or pattern.search(image.name)
    })
    if not target_domains:
        print(f"No validation file matches --select-pattern {args.select_pattern!r}; using all validation data.")
        target_domains = sorted(set(families["val"]))
    target_ids = [domain_index[domain] for domain in target_domains]

    print_source_table(domains, families, gsds, target_domains)
    normalization = gsd_normalization(gsds["train"])
    use_film = not args.no_film if stored is None else stored["architecture"]["kwargs"].get("use_film", True)
    if use_film and not args.evaluate_only:
        if not gsd_table.rows:
            print("WARNING: no --gsd-csv was given, so the GSD of every tile is unknown. FiLM then only ever "
                  "sees the null embedding and cannot learn anything from the GSD: this run trains the plain "
                  "U-Net with unused extra parameters. Pass --gsd-csv, or --no-film for the baseline.")
        elif normalization["known_tiles"] == 0:
            print("WARNING: no training file name matches a pattern of --gsd-csv, so the GSD of every "
                  "training tile is unknown and FiLM cannot learn anything from it. Check the patterns.")
        elif normalization["distinct_values"] < 2:
            print("WARNING: the training set has only one distinct GSD value. FiLM has no real scale "
                  "variation to learn from; the only variation left is what the scale augmentation "
                  "(rescale, affine, mosaic, ...) creates around that value.")
    if normalization["known_tiles"]:
        print(f"GSD known for {normalization['known_tiles']} of {len(splits['train'])} training tiles, "
              f"{normalization['distinct_values']} distinct value(s); log(GSD) mean "
              f"{normalization['log_mean']:.4f}, std {normalization['log_std']:.4f}.")

    datasets = {
        name: CrackSegmentationDataset(
            splits[name], [domain_index[family] for family in families[name]], gsds[name],
            training=name == "train", crop_size=args.image_size, options=options,
        )
        for name in ("train", "val", "test")
    }
    train_set = datasets["train"]
    enabled = augmentation["enabled_operations"]
    print(f"Augmentation preset: {args.aug_preset} | extended operations enabled: {len(enabled)}"
          + (f" ({', '.join(enabled)})" if 0 < len(enabled) <= 12 else "")
          + (f" | RandAugment N={options.randaugment[0]} M={options.randaugment[1]:g}" if options.randaugment else "")
          + (f" | label safety: {'on' if options.label_safety else 'off'}" if enabled or options.randaugment else ""))
    if args.preview_augmentations > 0:
        preview_folder = args.output / "augmentation_preview"
        shown = save_augmentation_preview(train_set, preview_folder, args.preview_augmentations, args.seed)
        print(f"Augmentation preview: {len(shown)} augmentation(s) on "
              f"{min(args.preview_augmentations, len(train_set))} tile(s) in {preview_folder}.")
        if args.preview_only:
            return

    pin = device.type == "cuda"
    loader_options = dict(num_workers=args.workers, pin_memory=pin, worker_init_fn=seed_worker,
                          persistent_workers=args.workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    sample_weights = [args.target_weight if family in target_domains else 1.0 for family in families["train"]]
    if args.target_weight != 1.0 and len(set(sample_weights)) > 1:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        train_loader = DataLoader(train_set, args.batch_size, sampler=sampler, drop_last=True, **loader_options)
    else:
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                                  generator=generator, **loader_options)
    # Evaluation memory is bounded twice: a batch holds at most --eval-batch-megapixels,
    # and an image with a side above --eval-tile is run as overlapping tiles.
    batch_pixels = args.eval_batch_megapixels * 1e6
    tiling = dict(tile=args.eval_tile, overlap=args.eval_tile_overlap,
                  tile_batch=max(1, int(batch_pixels // max(1, args.eval_tile) ** 2)))
    val_loader, test_loader = (
        DataLoader(datasets[name], batch_sampler=shape_grouped_batches(
            datasets[name].shapes, args.eval_batch_size, max_pixels=batch_pixels), **loader_options)
        for name in ("val", "test")
    )
    # The threshold is calibrated on the target-domain validation images only,
    # so the final passes do not need the other validation sources.
    target_set = set(target_ids)
    calibration_loader = DataLoader(datasets["val"], batch_sampler=shape_grouped_batches(
        datasets["val"].shapes, args.eval_batch_size,
        [index for index, domain in enumerate(datasets["val"].domain_ids) if domain in target_set],
        max_pixels=batch_pixels,
    ), **loader_options)
    if len(train_loader) == 0:
        raise SystemExit("Training set is smaller than one batch; lower --batch-size.")

    architecture = {"name": "hairline_film_unet", "kwargs": {
        "encoder_name": args.encoder, "decoder_channels": [256, 128, 64, 48, 32], "stem_channels": 32,
        "line_prior": not args.no_line_prior, "deep_supervision": not args.no_deep_supervision,
        "use_film": not args.no_film, "film_skips": args.film_skips and not args.no_film, "film_hidden": 64,
        "gsd_dropout": args.gsd_dropout,
        # The FiLM input is standardised with the training set's own log(GSD) statistics.
        "gsd_log_mean": normalization["log_mean"], "gsd_log_std": normalization["log_std"],
    }}
    resume_state = None
    if args.evaluate_only:  # rebuild exactly what was trained; its weights come from best.pt
        architecture = stored["architecture"]
    elif args.resume and last_path.is_file():
        resume_state = torch.load(last_path, map_location=device, weights_only=False)
        saved = resume_state["architecture"]["kwargs"]
        for key in ("gsd_log_mean", "gsd_log_std"):
            # The weights were trained with the saved normalisation; a changed CSV must not shift it.
            if not math.isclose(saved[key], architecture["kwargs"][key], rel_tol=1e-6, abs_tol=1e-9):
                print(f"Resuming with the {key} of last.pt ({saved[key]:.4f}), not the newly computed "
                      f"{architecture['kwargs'][key]:.4f}.")
            architecture["kwargs"][key] = saved[key]
    kwargs = architecture["kwargs"]
    model = HairlineFiLMUNet(pretrained=not (args.no_pretrained or args.evaluate_only), **kwargs)
    model = model.to(device, memory_format=torch.channels_last)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        parameter_groups(model, args.learning_rate, args.encoder_lr_factor, args.weight_decay)
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(
        args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch), args.min_lr_ratio
    ))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.loss == "focal":
        loss_function = FocalCrackLoss(args.focal_alpha, args.focal_gamma, args.cldice_weight,
                                       args.skeleton_iterations, args.aux_weights)
    else:
        loss_function = HairlineCrackLoss(
            args.bce_weight, args.tversky_weight, args.cldice_weight,
            args.false_positive_weight, args.false_negative_weight,
            args.skeleton_iterations, args.aux_weights,
        )
    thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)
    sweep = ThresholdSweep(thresholds, args.tolerance, len(domains), device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"Device: {device} | AMP: {use_amp} | parameters: {parameter_count:.1f} M | "
          f"train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}")
    if model.film is not None:
        film_count = sum(parameter.numel() for parameter in model.film.parameters())
        print(f"FiLM: on, {len(model.film.heads)} conditioned layers "
              f"({'decoder blocks and skips' if kwargs['film_skips'] else 'decoder blocks'}), "
              f"{film_count / 1e3:.1f} k parameters, GSD dropout {kwargs['gsd_dropout']:g}.")
    else:
        print("FiLM: off (--no-film): the plain hairline U-Net; GSD values are not used.")
    if train_set.contrast_factors is not None:
        factors = train_set.contrast_factors
        print(f"Contrast: all {len(factors)} levels from {100 * (factors[0] - 1):+.0f} % to "
              f"{100 * (factors[-1] - 1):+.0f} % in {100 * args.contrast_step:g} % steps; every training image "
              f"cycles through all of them (needs at least {len(factors)} epochs per image).")
    elif args.contrast_mode == "off" or not options.photometric:
        print("Contrast: unchanged.")
    else:
        print(f"Contrast: random change of {100 * args.contrast_min_change:g}-{100 * args.contrast_max_change:g} % "
              f"on {100 * args.contrast_p:g} % of the training images.")

    start_epoch, best_score, best_epoch, stale = 1, -1.0, 0, 0
    if resume_state is not None:
        state = resume_state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.module.load_state_dict(state["ema"])
            ema.updates = state["ema_updates"]
        start_epoch = state["epoch"] + 1
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale"]
        draws = state.get("contrast_draws")
        if (draws is not None and train_set.contrast_factors is not None
                and draws[0].shape == train_set.image_draws.shape and draws[1].shape == train_set.level_draws.shape):
            train_set.image_draws.copy_(draws[0])  # continue each image's cycle where it stopped
            train_set.level_draws.copy_(draws[1])
        fired = state.get("augmentation_fired")
        if fired is not None and fired[0].shape == train_set.fired.shape:
            train_set.fired.copy_(fired[0])  # the counts cover the whole run, not only this session
            train_set.safety.copy_(fired[1])
        print(f"Resumed from epoch {state['epoch']} (best {args.select_metric} {best_score:.4f}).")

    # What best.pt needs, besides the weights, to rebuild the model and to document the run.
    run_record = {
        "architecture": architecture, "use_film": kwargs["use_film"],
        "gsd_normalization": {**normalization, "log_mean": kwargs["gsd_log_mean"], "log_std": kwargs["gsd_log_std"]},
        "gsd_csv": [list(row) for row in gsd_table.rows], "augmentation": augmentation,
    }
    # History columns: train_*/val_* use the fixed threshold 0.5 so the two curves are
    # comparable; the unprefixed metrics are validation at that epoch's best threshold.
    fields = ["epoch", "lr", "train_loss", "val_loss",
              *(f"train_{name}" for name in BASIC_METRICS), *(f"val_{name}" for name in BASIC_METRICS),
              "threshold", *METRIC_NAMES, "all_val_f1", "all_val_tol_f1", "seconds"]
    history_path = args.output / "training_history.csv"
    resuming_history = args.resume and start_epoch > 1 and history_path.is_file()
    half_index = int(np.argmin(np.abs(thresholds - 0.5)))
    if not args.evaluate_only:
        with history_path.open("a" if resuming_history else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not resuming_history:
                writer.writeheader()
            for epoch in range(start_epoch, args.epochs + 1):
                started = time.time()
                model.train()
                cldice_scale = min(1.0, epoch / max(1, args.cldice_warmup_epochs))
                running_loss, seen = 0.0, 0
                train_counts = torch.zeros(4, dtype=torch.float64, device=device)  # tp, fp, fn, pixels
                for images, masks, valid, _, _, gsd in train_loader:
                    images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    masks, valid = masks.to(device, non_blocking=True), valid.to(device, non_blocking=True)
                    gsd = gsd.to(device, non_blocking=True)  # already divided by each tile's augmentation scale
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        outputs = model(images, gsd)
                    loss = loss_function(outputs, masks, valid, cldice_scale)
                    if not torch.isfinite(loss):
                        print("Warning: non-finite loss; batch skipped.")
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
                    running_loss += loss.item() * images.size(0)
                    seen += images.size(0)
                    with torch.no_grad():  # training metrics at threshold 0.5 (logit 0)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        supervised = valid > 0.5
                        prediction, truth = (logits.detach() >= 0) & supervised, (masks >= 0.5) & supervised
                        train_counts += torch.stack((
                            (prediction & truth).sum(), (prediction & ~truth).sum(),
                            (~prediction & truth).sum(), supervised.sum(),
                        )).double()

                evaluation_model = ema.module if ema is not None else model
                val_loss = evaluate(evaluation_model, val_loader, device, sweep, use_amp, loss_function, **tiling)
                target_metrics, all_metrics = sweep.metrics(target_ids), sweep.metrics()
                best_index = int(np.argmax(target_metrics[args.select_metric]))
                score = float(target_metrics[args.select_metric][best_index])
                train_metrics = basic_metrics(*train_counts.tolist())
                row = {
                    "epoch": epoch, "lr": max(group["lr"] for group in optimizer.param_groups),
                    "train_loss": running_loss / max(1, seen), "val_loss": val_loss,
                    **{f"train_{name}": train_metrics[name] for name in BASIC_METRICS},
                    **{f"val_{name}": float(target_metrics[name][half_index]) for name in BASIC_METRICS},
                    "threshold": float(thresholds[best_index]), **at_threshold(target_metrics, best_index),
                    "all_val_f1": float(all_metrics["f1"].max()),
                    "all_val_tol_f1": float(all_metrics["tol_f1"].max()),
                    "seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                improved = score > best_score
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | loss {row['train_loss']:.4f}/{val_loss:.4f} | "
                    f"thr {row['threshold']:.3f} P {row['precision']:.4f} R {row['recall']:.4f} "
                    f"F1 {row['f1']:.4f} IoU {row['iou']:.4f} | tolF1 {row['tol_f1']:.4f} | "
                    f"{row['seconds']:.0f}s{'  *best*' if improved else ''}"
                )
                if improved:
                    best_score, best_epoch, stale = score, epoch, 0
                    torch.save({
                        "model": evaluation_model.state_dict(), **run_record,
                        "arguments": json_safe(vars(args)), "epoch": epoch,
                        "best_val_score": best_score, "select_metric": args.select_metric,
                        "optimal_threshold": float(thresholds[best_index]),
                    }, best_path)
                else:
                    stale += 1
                torch.save({
                    "model": model.state_dict(), "ema": ema.module.state_dict() if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best_score,
                    "best_epoch": best_epoch, "stale": stale, "architecture": architecture,
                    "arguments": json_safe(vars(args)),
                    "contrast_draws": None if train_set.contrast_factors is None else
                    [train_set.image_draws.clone(), train_set.level_draws.clone()],
                    "augmentation_fired": [train_set.fired.clone(), train_set.safety.clone()],
                }, last_path)
                if stale >= args.patience:
                    print(f"Early stopping: no {args.select_metric} improvement for {args.patience} epochs.")
                    break
        coverage = train_set.contrast_coverage()
        if coverage is not None:
            (args.output / "contrast_coverage.json").write_text(json.dumps(coverage, indent=2))
            print(f"Contrast coverage: {coverage['images_trained_at_every_level']} of "
                  f"{coverage['training_images']} training images were trained at all {coverage['levels']} "
                  f"levels (draws per image: minimum {coverage['minimum_draws_per_image']}, "
                  f"mean {coverage['mean_draws_per_image']:.1f}).")
        counts = train_set.augmentation_counts()
        (args.output / "augmentation_counts.json").write_text(json.dumps(counts, indent=2))
        safety = counts["label_safety"]
        print(f"Augmentation counts: {counts['training_samples_drawn']} training samples drawn; label safety "
              f"excluded {safety['crack_pixels_excluded']} of {safety['crack_pixels_checked']} checked crack "
              f"pixels in {safety['checks']} checks.")

    # ---------------- Calibration on validation, then a single test pass ---------------- #
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"\nBest epoch {checkpoint['epoch']} ({args.select_metric} {checkpoint['best_val_score']:.4f}).")
    use_tta = not args.no_tta
    calibration, evaluation = {}, {}
    for tta in sorted({False, use_tta}):
        evaluate(model, calibration_loader, device, sweep, use_amp, tta=tta, **tiling)
        validation_counts = sweep.count_array(target_ids)
        metrics = metrics_from_counts(validation_counts)
        index = int(np.argmax(metrics[args.select_metric]))
        stored_threshold = checkpoint.get("optimal_threshold_tta" if tta else "optimal_threshold")
        if args.evaluate_only and stored_threshold is not None:
            # Report at the threshold the training run calibrated (and the detector uses).
            # Recalibrating on another machine can move it by a grid step, because the
            # validation curve is flat near its maximum and CPU/GPU precision differs.
            index = int(np.argmin(np.abs(thresholds - stored_threshold)))
            print(f"Threshold stored in best.pt ({'flip TTA' if tta else 'no TTA'}): {thresholds[index]:.3f}")
        else:
            print(f"Validation-calibrated threshold ({'flip TTA' if tta else 'no TTA'}): {thresholds[index]:.3f}")
        calibration[tta] = (index, metrics)
        evaluation["tta" if tta else "no_tta"] = {"threshold_index": index, "val": validation_counts}
    if not args.evaluate_only:
        with (args.output / "val_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            sweep_writer = csv.writer(handle)
            sweep_writer.writerow(["tta", "threshold", *METRIC_NAMES])
            for tta, (_, metrics) in calibration.items():
                for index, threshold in enumerate(thresholds):
                    sweep_writer.writerow(
                        [tta, f"{threshold:.3f}", *(f"{metrics[name][index]:.5f}" for name in METRIC_NAMES)]
                    )
        checkpoint["optimal_threshold"] = float(thresholds[calibration[False][0]])
        checkpoint["optimal_threshold_tta"] = float(thresholds[calibration[use_tta][0]])
        checkpoint["threshold_metrics"] = {
            str(tta): {name: calibration[tta][1][name].tolist() for name in METRIC_NAMES} for tta in calibration
        }
        checkpoint["thresholds"] = thresholds.tolist()
        torch.save(checkpoint, best_path)

    results = {
        "best_epoch": checkpoint["epoch"], "select_metric": args.select_metric,
        "best_validation_score": checkpoint["best_val_score"], "tolerance_px": args.tolerance,
        "target_sources": target_domains,
    }
    # The same pixel counts as per source, grouped by the GSD of the test tiles instead.
    test_set = datasets["test"]
    gsd_labels, gsd_group_ids = gsd_groups(test_set.gsds)
    gsd_sweep = ThresholdSweep(thresholds, args.tolerance, len(gsd_labels), device)
    for tta in sorted(calibration):
        index, mode = calibration[tta][0], "tta" if tta else "no_tta"
        hook = None
        if tta == use_tta and args.save_predictions > 0 and not args.evaluate_only:
            hook = prediction_writer(test_set, args.output / "test_predictions",
                                     args.save_predictions, float(thresholds[index]))
        test_loss = evaluate(model, test_loader, device, sweep, use_amp, loss_function, tta=tta, hook=hook,
                             gsd_sweep=gsd_sweep, gsd_group_ids=torch.tensor(gsd_group_ids), **tiling)
        evaluation[mode]["test"] = sweep.count_array()
        per_source = {domains[d]: at_threshold(sweep.metrics([d]), index)
                      for d in sorted(set(test_set.domain_ids))}
        per_gsd = {gsd_labels[group]: {"images": gsd_group_ids.count(group),
                                       **at_threshold(gsd_sweep.metrics([group]), index)}
                   for group in sorted(set(gsd_group_ids))}
        results[mode] = {
            "threshold": float(thresholds[index]), "loss": test_loss,
            **at_threshold(sweep.metrics(), index), "per_source": per_source, "per_gsd": per_gsd,
        }
    if not args.evaluate_only:  # an evaluation-only run never overwrites the training run's files
        (args.output / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print("Test:", json.dumps(results, indent=2))

    # ---------------- GSD sanity check: does the model use the GSD? ---------------- #
    plot_folder = args.output / "plots"
    sanity = None
    if args.gsd_sanity_check and model.film is None:
        print("GSD sanity check skipped: a --no-film model has no GSD input, so every pass would be identical.")
    elif args.gsd_sanity_check:
        sanity = gsd_sanity_check(model, test_loader, device, sweep, use_amp, use_tta,
                                  calibration[use_tta][0], **tiling)
        sanity["test_tiles"] = len(test_set)
        sanity["test_tiles_with_known_gsd"] = sum(math.isfinite(gsd) for gsd in test_set.gsds)
        (args.output / "gsd_sanity_check.json").write_text(json.dumps(sanity, indent=2))
        if sanity["test_tiles_with_known_gsd"] == 0:
            print("Note: no test tile has a known GSD, so all four passes used the null embedding and are "
                  "identical. The check says nothing without GSD values.")
        elif max(sanity["variants"][0][key] for key in GSD_SANITY_METRICS) == 0:
            print("Note: the model detects nothing on the test set at this threshold, so the check says "
                  "nothing about its use of the GSD yet.")
        elif sanity["ignores_gsd"]:
            print(f"Note: F1, tolerance F1 and IoU change by at most {sanity['largest_change']:.4f} when the "
                  "GSD is halved, doubled or withheld. The model appears to ignore the GSD.")
        else:
            print(f"The metrics change by up to {sanity['largest_change']:.4f} with the GSD input: the model "
                  "uses it.")

    # ---------------- Tables and figures ---------------- #
    save_evaluation_tables(plot_folder, thresholds, evaluation)
    try:
        written = save_plots(args.output, thresholds, evaluation, checkpoint["epoch"], args.tolerance)
        if sanity is not None:
            written.append(save_gsd_sanity_plot(plot_folder, sanity, args.tolerance))
        print(f"Figures in {plot_folder}: {', '.join(written)}")
    except ImportError:
        print("matplotlib is not installed, so no figures were drawn (python -m pip install matplotlib). "
              f"The curve and confusion-matrix tables are in {plot_folder}.")


if __name__ == "__main__":
    main()

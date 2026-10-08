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

This is the server version of ``train_unet_film_hairline.py``. The functions
and classes of that script are unchanged and split over the modules of
``filmunet``; the workflow of the Swin U-Net server code is added to them (see
"Server version" below).

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
* ``--encoder swin_t`` (or another ``swin_*``) builds ``HairlineFiLMSwinUNet``
  instead: the Swin-Unet of the Swin U-Net project (Swin encoder, three Swin
  up-stages at 1/16, 1/8 and 1/4 scale, then the same two convolutional
  blocks and thin-line stem at 1/2 and 1/1 scale) with the same FiLM
  conditioning on its five decoder stages. It trains with that project's
  optimiser defaults: learning rate 2e-4, encoder factor 0.3, weight decay
  0.01.

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

Server version (added to ``train_unet_film_hairline.py``, as in the Swin U-Net
server code):

* ``last.pt`` and ``best.pt`` are written to a temporary file and renamed, so
  a job killed while saving leaves the previous checkpoint intact, and a
  resumed run does not log a repeated epoch twice.
* ``validation_by_source.csv`` records every validation source at every epoch
  and the log prints the three weakest. The pooled metrics are summed over
  pixels, so the thick-crack sources dominate them and can hide a source the
  model fails on.
* ``--init-weights best.pt`` fine-tunes an earlier model, with or without
  FiLM, and ``--label-tolerance N`` leaves an N-pixel ring around every label
  out of the loss, for labels drawn a few pixels off the crack.
* The test overlays are spread evenly over the sources.

``--self-test`` checks the FiLM identity at initialisation, the alignment,
binary labels and GSD bookkeeping of every augmentation, and the unknown-GSD
path through dataset, augmentation and model, then exits.
"""

from __future__ import annotations
from .common import (
    BASIC_METRICS, COUNT_NAMES, Counter, DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_PATH, DataLoader, ENCODER_CHANNELS,
    METRIC_NAMES, Path, SWIN_ENCODERS, WeightedRandomSampler, argparse, csv, defaultdict, json, math, nn, np, os,
    random, re, seed_everything, seed_worker, sys, time, torch,
)
from .gsd import GSDTable, gsd_groups, gsd_normalization, print_source_table
from .augmentations import (
    AUGMENTATIONS, AugmentationOptions, LABEL_SAFETY_CELL, LABEL_SAFETY_GREY_LEVELS, NO_AUGMENTATION,
    PRESET_MAGNITUDE, PRESET_MAX_ROTATION, PRESET_NAMES, RANDAUGMENT_POOL, preset_probabilities,
)
from .data import (
    CrackSegmentationDataset, collect_pairs, save_augmentation_preview, shape_grouped_batches, source_family,
)
from .models import ARCHITECTURES, ModelEMA
from .losses import FocalCrackLoss, HairlineCrackLoss
from .evaluation import (
    GSD_SANITY_METRICS, ThresholdSweep, at_threshold, basic_metrics, evaluate, gsd_sanity_check,
    metrics_from_counts, prediction_writer,
)
from .plots import save_evaluation_tables, save_gsd_sanity_plot, save_plots

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
            # Nor on a Swin encoder's relative position terms, as in the Swin U-Net project.
            no_decay = parameter.ndim <= 1 or name.startswith("film.") or any(
                key in name for key in ("relative_position_bias_table", "logit_scale", "cpb_mlp")
            )
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


def save_checkpoint(state, path: Path) -> None:
    """Write to a temporary file, then rename it over `path`.

    A job killed while saving (time limit, scancel) then leaves the previous
    checkpoint intact instead of a truncated file that cannot be resumed.
    """
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def open_epoch_log(path: Path, fields, first_epoch: int):
    """Open a per-epoch CSV (epoch in the first column) for the rows from `first_epoch` on.

    A fresh run starts the file. A resumed run keeps the rows of the epochs before
    `first_epoch` and drops later ones: a job killed between writing an epoch's
    row and saving last.pt repeats that epoch, which would otherwise be logged twice.
    """
    rows = []
    if first_epoch > 1 and path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        rows = rows[:1] + [row for row in rows[1:] if row and row[0].isdigit() and int(row[0]) < first_epoch]
    handle = path.open("w", newline="", encoding="utf-8")
    csv.writer(handle).writerows(rows or [list(fields)])
    handle.flush()
    return handle


def keep_gsd_normalization(kwargs: dict, saved: dict, source: str) -> None:
    """The weights were trained with the saved normalisation; a changed CSV must not shift it."""
    for key in ("gsd_log_mean", "gsd_log_std"):
        if not math.isclose(saved[key], kwargs[key], rel_tol=1e-6, abs_tol=1e-9):
            print(f"Using the {key} of {source} ({saved[key]:.4f}), not the newly computed {kwargs[key]:.4f}.")
        kwargs[key] = saved[key]


def initial_weights(path: Path, kwargs: dict) -> dict:
    """The best.pt of an earlier run for --init-weights, checked against the network this run builds.

    The network must be the same, FiLM aside. Weights trained without FiLM get this run's
    generator, which starts as the identity, so training starts exactly from that model.
    Weights trained with FiLM need the same FiLM layers, and their log(GSD) normalisation is kept.
    """
    initial = torch.load(path, map_location="cpu", weights_only=False)
    saved = initial["architecture"]["kwargs"]
    network = ("encoder_name", "decoder_channels", "decoder_depth", "head_channels", "stem_channels",
               "line_prior", "deep_supervision")
    different = {key: saved[key] for key in network if key in saved and key in kwargs and saved[key] != kwargs[key]}
    if different:
        raise SystemExit(f"--init-weights holds a different network ({different}); this run builds "
                         f"{ {key: kwargs[key] for key in different} }. Pass the matching model options.")
    if saved.get("use_film"):
        if not kwargs["use_film"] or bool(saved.get("film_skips")) != kwargs["film_skips"]:
            raise SystemExit("--init-weights was trained with FiLM" + (" on the skips (--film-skips)"
                             if saved.get("film_skips") else " on the decoder only") + "; this run must use it too.")
        keep_gsd_normalization(kwargs, saved, path.name)
    return initial


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


# --learning-rate, --encoder-lr-factor and --weight-decay where they are not given.
OPTIMISER_DEFAULTS = {"resnet": (3e-4, 0.1, 1e-4), "swin": (2e-4, 0.3, 0.01)}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET_PATH,
                        help=f"80/10/10 dataset root (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resume", action="store_true", help="Continue from <output>/last.pt")
    parser.add_argument("--init-weights", type=Path,
                        help="Start from the weights of an earlier run's best.pt instead of ImageNet, to "
                             "fine-tune on new data. A model trained without FiLM (--no-film, "
                             "train_unet_hairline.py, or the Swin U-Net project for a Swin encoder) gets a "
                             "FiLM generator that starts as the identity. The run itself is fresh (new "
                             "schedule, EMA and history); not used when --resume continues from last.pt")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="No training: evaluate <output>/best.pt on validation and test and write the "
                             "figures, curves and confusion matrices to <output>/plots (other files untouched)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the built-in checks (FiLM identity, augmentation alignment and GSD "
                             "bookkeeping, unknown GSD) on a temporary dummy dataset, then exit")
    # Model
    parser.add_argument("--encoder", choices=(*ENCODER_CHANNELS, *SWIN_ENCODERS), default="resnet34",
                        help="resnet*: the FiLM U-Net (default resnet34). swin_*: the same conditioning on "
                             "the Swin-Unet of the Swin U-Net project")
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
    # CNN settings (those of train_unet_hairline.py) for a ResNet encoder; the transformer
    # fine-tuning ones of the Swin U-Net project for a Swin encoder (OPTIMISER_DEFAULTS).
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw",
                        help="adamw: decoupled weight decay (default, as the reference). adam: plain Adam, "
                             "where --weight-decay acts as L2 regularisation")
    parser.add_argument("--learning-rate", type=float, help="Default: 3e-4 (ResNet), 2e-4 (Swin)")
    parser.add_argument("--encoder-lr-factor", type=float, help="Default: 0.1 (ResNet), 0.3 (Swin)")
    parser.add_argument("--weight-decay", type=float, help="Default: 1e-4 (ResNet), 0.01 (Swin)")
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
    parser.add_argument("--label-tolerance", type=int, default=0,
                        help="Leave the ring of this many pixels around every label out of the loss: "
                             "hand-drawn labels sit a few pixels off the crack, and the tol_* metrics "
                             "already accept a prediction that close. 0 = strict loss (default)")
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
    parser.add_argument("--save-predictions", type=int, default=40,
                        help="Test overlays to save, spread evenly over the sources; a number at least "
                             "the size of the test set saves every image")
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
    defaults = OPTIMISER_DEFAULTS["swin" if args.encoder in SWIN_ENCODERS else "resnet"]
    for name, value in zip(("learning_rate", "encoder_lr_factor", "weight_decay"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.image_size % 32:
        parser.error("--image-size must be divisible by 32 (the encoder halves the image five times).")
    if args.eval_tile % 32 or not 0 <= args.eval_tile_overlap < max(1, args.eval_tile):
        parser.error("--eval-tile must be divisible by 32 and larger than --eval-tile-overlap.")
    if args.eval_batch_megapixels <= 0:
        parser.error("--eval-batch-megapixels must be greater than zero.")
    if not 0 <= args.gsd_dropout <= 1:
        parser.error("--gsd-dropout must lie in [0, 1].")
    if args.label_tolerance < 0:
        parser.error("--label-tolerance must not be negative.")
    if args.init_weights is not None and not args.evaluate_only and not args.init_weights.is_file():
        parser.error(f"--init-weights file not found: {args.init_weights}")
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


def main():
    args = parse_arguments()
    if args.self_test:
        from .self_test import run_self_tests  # here, because self_test imports this module

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

    swin = args.encoder in SWIN_ENCODERS
    architecture = {"name": "hairline_film_swin_unet" if swin else "hairline_film_unet", "kwargs": {
        "encoder_name": args.encoder,
        **({"decoder_depth": 2, "head_channels": [48, 32]} if swin else {"decoder_channels": [256, 128, 64, 48, 32]}),
        "stem_channels": 32,
        "line_prior": not args.no_line_prior, "deep_supervision": not args.no_deep_supervision,
        "use_film": not args.no_film, "film_skips": args.film_skips and not args.no_film, "film_hidden": 64,
        "gsd_dropout": args.gsd_dropout,
        # The FiLM input is standardised with the training set's own log(GSD) statistics.
        "gsd_log_mean": normalization["log_mean"], "gsd_log_std": normalization["log_std"],
    }}
    resume_state, initial = None, None
    if args.evaluate_only:  # rebuild exactly what was trained; its weights come from best.pt
        architecture = stored["architecture"]
    elif args.resume and last_path.is_file():
        resume_state = torch.load(last_path, map_location=device, weights_only=False)
        keep_gsd_normalization(architecture["kwargs"], resume_state["architecture"]["kwargs"], "last.pt")
    elif args.init_weights is not None:
        initial = initial_weights(args.init_weights, architecture["kwargs"])
    kwargs = architecture["kwargs"]
    model = ARCHITECTURES[architecture["name"]](
        pretrained=not (args.no_pretrained or args.evaluate_only or initial is not None), **kwargs)
    if initial is not None:
        keys = model.load_state_dict(initial["model"], strict=False)
        unfit = [key for key in keys.missing_keys if not key.startswith("film.")] + keys.unexpected_keys
        if unfit:
            raise SystemExit(f"--init-weights does not fit this network: {len(unfit)} parameters differ, "
                             f"for example {unfit[:3]}.")
        print(f"Initial weights: {args.init_weights} (epoch {initial.get('epoch')}, "
              f"{initial.get('select_metric')} {initial.get('best_val_score', float('nan')):.4f})"
              + ("; the FiLM generator is new and starts as the identity." if keys.missing_keys else "."))
        del initial
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
                                       args.skeleton_iterations, args.aux_weights, args.label_tolerance)
    else:
        loss_function = HairlineCrackLoss(
            args.bce_weight, args.tversky_weight, args.cldice_weight,
            args.false_positive_weight, args.false_negative_weight,
            args.skeleton_iterations, args.aux_weights, args.label_tolerance,
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
        print("FiLM: off (--no-film): the plain hairline network; GSD values are not used.")
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
    # One row per validation source and epoch, at that epoch's selected threshold; best_* is
    # the source's own optimum. The pooled history hides a source the model fails on.
    source_path = args.output / "validation_by_source.csv"
    source_fields = ["epoch", "source", "images", "target", "threshold", *METRIC_NAMES,
                     "best_threshold", f"best_{args.select_metric}"]
    val_sources, val_images = sorted(set(datasets["val"].domain_ids)), Counter(families["val"])
    labelled_column = COUNT_NAMES.index("labelled")
    half_index = int(np.argmin(np.abs(thresholds - 0.5)))
    if not args.evaluate_only:
        with open_epoch_log(history_path, fields, start_epoch) as handle, \
                open_epoch_log(source_path, source_fields, start_epoch) as source_handle:
            writer, source_writer = csv.DictWriter(handle, fieldnames=fields), csv.writer(source_handle)
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
                source_scores = []  # (score at the selected threshold, name) of target sources with labels
                for domain_id in val_sources:
                    name, source_counts = domains[domain_id], sweep.count_array([domain_id])
                    source_metrics = metrics_from_counts(source_counts)
                    own_index = int(np.argmax(source_metrics[args.select_metric]))
                    source_writer.writerow([
                        epoch, name, val_images[name], int(name in target_domains),
                        f"{thresholds[best_index]:.3f}",
                        *(f"{source_metrics[metric][best_index]:.5f}" for metric in METRIC_NAMES),
                        f"{thresholds[own_index]:.3f}", f"{source_metrics[args.select_metric][own_index]:.5f}",
                    ])
                    if name in target_domains and source_counts[0, labelled_column] > 0:
                        source_scores.append((float(source_metrics[args.select_metric][best_index]), name))
                source_handle.flush()
                improved = score > best_score
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | loss {row['train_loss']:.4f}/{val_loss:.4f} | "
                    f"thr {row['threshold']:.3f} P {row['precision']:.4f} R {row['recall']:.4f} "
                    f"F1 {row['f1']:.4f} IoU {row['iou']:.4f} | tolF1 {row['tol_f1']:.4f} | "
                    f"{row['seconds']:.0f}s{'  *best*' if improved else ''}"
                )
                if len(source_scores) > 1:
                    print(f"    weakest sources ({args.select_metric}): "
                          + " | ".join(f"{name} {value:.3f}" for value, name in sorted(source_scores)[:3]))
                if improved:
                    best_score, best_epoch, stale = score, epoch, 0
                    save_checkpoint({
                        "model": evaluation_model.state_dict(), **run_record,
                        "arguments": json_safe(vars(args)), "epoch": epoch,
                        "best_val_score": best_score, "select_metric": args.select_metric,
                        "optimal_threshold": float(thresholds[best_index]),
                    }, best_path)
                else:
                    stale += 1
                save_checkpoint({
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
        save_checkpoint(checkpoint, best_path)

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


__all__ = [
    'parameter_groups', 'warmup_cosine', 'json_safe', 'save_checkpoint', 'open_epoch_log',
    'keep_gsd_normalization', 'initial_weights', 'flag_name', 'REFERENCE_AUGMENTATION_ARGUMENTS',
    'resolve_augmentation', 'OPTIMISER_DEFAULTS', 'parse_arguments', 'check_augmentation_arguments', 'main',
]

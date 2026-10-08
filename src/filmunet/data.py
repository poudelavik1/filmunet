'Data for hairline crack segmentation.'

from __future__ import annotations
from .common import Dataset, IMAGE_EXTENSIONS, cv2, defaultdict, math, np, random, re, torch
from .gsd import UNKNOWN_GSD, format_gsd, gsd_key
from .augmentations import (
    AUGMENTATIONS, AugmentationOptions, LABEL_SAFETY_GREY_LEVELS, PIPELINE_ORDER, RANDAUGMENT_POOL,
    STAGE_OPERATIONS, Tile, add_synthetic_hairline, contrast_level_factors, cutout, dihedral, elastic,
    fade_cracks, label_safety_check, local_background, photometric_augment, random_affine, random_crop,
    reference_blur, reference_jpeg, reference_noise, reference_tone, rescale, resize_tile,
)

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


__all__ = [
    'normalized_stem', 'tile_coordinate_key', 'unique_map', 'collect_pairs', 'source_family', 'image_shape',
    'read_pair', 'CrackSegmentationDataset', 'preview_panel', 'save_augmentation_preview',
    'shape_grouped_batches',
]

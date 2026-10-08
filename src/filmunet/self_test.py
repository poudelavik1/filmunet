'Built-in checks of the trainer (--self-test).'

from __future__ import annotations
from .common import DataLoader, Path, cv2, math, np, random, seed_everything, torch
from .gsd import GSDTable, UNKNOWN_GSD, gsd_key
from .augmentations import (
    AUGMENTATIONS, AugmentationOptions, PIPELINE_ORDER, STAGES, Tile, dihedral, elastic, label_safety_check,
    preset_probabilities, random_affine, random_crop, rescale, stage_names, to_uint8,
)
from .data import CrackSegmentationDataset, collect_pairs, save_augmentation_preview, source_family
from .models import HairlineFiLMUNet
from .losses import FocalCrackLoss, HairlineCrackLoss
from .evaluation import predict_logits
from .training import __doc__ as TRAINER_DOCUMENTATION, parameter_groups

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

    print("Preset table of the trainer's docstring")
    documented = {}
    for line in TRAINER_DOCUMENTATION.splitlines():
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


__all__ = [
    'SELF_TEST_CUT', 'SCALE_CHANGING', 'GEOMETRIC_STAGES', 'encoded_tile', 'write_self_test_dataset',
    'run_self_tests',
]

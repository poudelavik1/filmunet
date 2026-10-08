'Reference augmentation and the extended augmentation library.'

from __future__ import annotations
from .common import cv2, dataclass, field, math, np, random

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


__all__ = [
    'AugmentationOptions', 'rescale', 'pad_to_at_least', 'random_crop', 'dihedral', 'random_affine', 'elastic',
    'local_background', 'fade_cracks', 'add_synthetic_hairline', 'cutout', 'motion_blur', 'adjust_contrast',
    'random_contrast_factor', 'contrast_level_factors', 'reference_tone', 'reference_blur', 'reference_noise',
    'reference_jpeg', 'photometric_augment', 'Tile', 'Augmentation', 'STAGES', 'RANDAUGMENT_STAGES',
    'AUGMENTATIONS', 'register', 'amount', 'signed', 'to_uint8', 'resize_tile', 'warp_matrix', 'warp_maps',
    'pixel_grid', 'smooth_noise', 'soft_blobs', 'line_distance', 'straight_line', 'line_through', 'line_alpha',
    'blend', 'exclude_hidden_cracks', 'LABEL_SAFETY_GREY_LEVELS', 'LABEL_SAFETY_CELL', 'label_safety_check',
    'mosaic', 'cutmix', 'mixup', 'meander', 'paint_crack', 'branch_crack', 'varwidth_crack', 'construction_joint',
    'saw_cut', 'scratches', 'painted_line', 'cable_shadow', 'stain', 'efflorescence', 'dirt', 'spall',
    'resized_crop', 'zoom_out', 'shear', 'translate', 'homography_scale', 'perspective', 'optical_distortion',
    'distorted_axis', 'grid_distortion', 'piecewise_affine', 'stitching_seam', 'nodata_border', 'exposure_step',
    'blur_patch', 'ghosting', 'LUMA', 'rgb_shift', 'channel_shuffle', 'color_jitter', 'blackbody_rgb',
    'white_balance', 'tone_curve', 'autocontrast', 'equalize', 'sharpen', 'emboss', 'posterize', 'solarize',
    'overexposure', 'underexposure', 'cast_shadow', 'radial_distance', 'sun_flare', 'vignette', 'spotlight',
    'fog', 'wet_surface', 'rain', 'defocus_blur', 'median_blur', 'zoom_blur', 'glass_blur', 'poisson_noise',
    'iso_noise', 'speckle_noise', 'salt_pepper', 'pixel_dropout', 'chromatic_aberration', 'ringing',
    'webp_compression', 'downscale_upscale', 'hide', 'random_erasing', 'gridmask', 'hide_and_seek',
    'STAGE_OPERATIONS', 'stage_names', 'RANDAUGMENT_POOL', 'PIPELINE_ORDER', 'PRESET_NAMES',
    'PRESET_PROBABILITIES', 'ALL_PRESET_PROBABILITY', 'PRESET_MAGNITUDE', 'PRESET_MAX_ROTATION',
    'NO_AUGMENTATION', 'preset_probabilities',
]

'Ground sampling distance (GSD, mm per pixel) for hairline crack segmentation.'

from __future__ import annotations
from .common import Path, csv, math, np, re

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


__all__ = [
    'UNKNOWN_GSD', 'GSD_LOG_STD_FLOOR', 'GSDTable', 'gsd_key', 'format_gsd', 'describe_gsd', 'gsd_normalization',
    'gsd_groups', 'print_source_table',
]

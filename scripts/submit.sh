#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/submit.sh /absolute/dataset/path [output-directory]" >&2
    exit 2
fi
DATASET="$(realpath -e "$1")" || { echo "Dataset path does not exist: $1" >&2; exit 1; }
export DATASET
for split in train val test; do
    for kind in images masks; do
        test -d "$DATASET/$split/$kind" || { echo "Missing $DATASET/$split/$kind" >&2; exit 1; }
    done
done
if [[ -n "${SELECT_PATTERN:-}" ]]; then
    # The trainer silently falls back to all sources when nothing matches, so check before queueing.
    matched=$(ls "$DATASET/val/images" | grep -cE -- "$SELECT_PATTERN" || true)
    [[ "$matched" -gt 0 ]] || { echo "SELECT_PATTERN matches no validation image: $SELECT_PATTERN" >&2; exit 1; }
    echo "Target: $matched validation images match $SELECT_PATTERN"
fi
if [[ -n "${INIT_WEIGHTS:-}" ]]; then
    # Resolve before the cd below, and fail here rather than after the queue wait.
    INIT_WEIGHTS="$(realpath -e "$INIT_WEIGHTS")" || { echo "INIT_WEIGHTS does not exist" >&2; exit 1; }
    export INIT_WEIGHTS
fi
if [[ "${NO_FILM:-0}" != 1 ]]; then
    # Without GSD values the trainer only warns and FiLM learns nothing, so check the table before queueing.
    GSD_CSV="$(realpath -e "${GSD_CSV:-$ROOT/configs/gsd.csv}")" || { echo "GSD_CSV does not exist" >&2; exit 1; }
    export GSD_CSV
    known=0
    while IFS= read -r row; do
        pattern="${row%,*}"
        count=$(ls "$DATASET/train/images" | grep -cE -- "$pattern" || true)
        printf 'GSD %s mm/px: %s training images match %s\n' "${row##*,}" "$count" "$pattern"
        known=$((known + count))
    done < <(tail -n +2 "$GSD_CSV" | tr -d '\r' | grep -vE '^[[:space:]]*(#|$)')
    [[ "$known" -gt 0 ]] || { echo "No training image matches a pattern of $GSD_CSV" >&2; exit 1; }
fi
ENCODER="${ENCODER:-resnet34}"
export ENCODER
if [[ "${RESUME:-0}" != 1 && -z "${INIT_WEIGHTS:-}" ]]; then
    # Compute nodes may lack internet, so the pretrained encoder must already be in the cache.
    compgen -G "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/$ENCODER-*.pth" >/dev/null || {
        echo "Pretrained $ENCODER weights are not cached. Run: ENCODER=$ENCODER bash scripts/setup_conda.sh" >&2
        exit 1
    }
fi
cd "$ROOT"
mkdir -p logs runs
export OUTPUT_DIR="$(realpath -m "${2:-$ROOT/runs/$(date +%Y%m%d_%H%M%S)_$$}")"
if [[ "${RESUME:-0}" == 1 ]]; then
    test -f "$OUTPUT_DIR/last.pt" || { echo "No last.pt to resume" >&2; exit 1; }
elif [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists. Choose a new directory or use RESUME=1." >&2
    exit 1
fi
# mkdir belongs to the allocated job, so a failed submission leaves no results folder.
submission=(--parsable --export=ALL)
[[ -z "${SLURM_ACCOUNT:-}" ]] || submission+=(--account="$SLURM_ACCOUNT")
[[ -z "${SLURM_PARTITION:-}" ]] || submission+=(--partition="$SLURM_PARTITION")
raw_job=$(sbatch "${submission[@]}" slurm/train_film.sbatch)
job_id="${raw_job%%;*}"
printf 'Job: %s\nResults: %s\n' "$job_id" "$OUTPUT_DIR"
printf 'Watch log: tail -F logs/film_unet_%s.out\n' "$job_id"
printf 'Watch metrics: python scripts/monitor_epochs.py "%s" --watch\n' "$OUTPUT_DIR"

# FiLM U-Net hairline crack segmentation architecture

Organization follows the Swin U-Net project (configs, src, scripts, slurm, docs),
which follows https://github.com/Samir4456/pgat-length. The model, augmentations,
losses, metrics, and training logic come from `train_unet_film_hairline.py`.

| Module | Responsibility |
| --- | --- |
| `src/filmunet/common.py` | Shared imports, constants, default paths, random seeds |
| `src/filmunet/gsd.py` | GSD table (`pattern,gsd_mm_per_px`), log(GSD) normalisation, per-GSD groups |
| `src/filmunet/augmentations.py` | Reference augmentation, extended library, label safety, presets |
| `src/filmunet/data.py` | Image/mask pairing, dataset with GSD bookkeeping, augmentation preview, batching |
| `src/filmunet/models.py` | FiLM generator, FiLM U-Net (ResNet), FiLM Swin-Unet, EMA, checkpoint reconstruction |
| `src/filmunet/losses.py` | BCE, Tversky, clDice, focal loss |
| `src/filmunet/evaluation.py` | Threshold sweeps, tiled inference, tolerant metrics, GSD sanity check, predictions |
| `src/filmunet/plots.py` | Epoch curves and final evaluation figures/tables |
| `src/filmunet/training.py` | Argument parsing, optimizer/scheduler, training/validation, resume |
| `src/filmunet/self_test.py` | The trainer's built-in checks (`--self-test`) |
| `scripts/train.py` | TOML defaults and training CLI |
| `slurm/train_film.sbatch` | ASL GPU allocation and execution |

The functions and classes of `train_unet_film_hairline.py` were extracted without
changing them. Eight were then edited for the additions below; the rest are
identical to the script, which stays in the project root for reference.

## Added to the script

Server workflow, taken from the Swin U-Net project:

- `last.pt` and `best.pt` are written to a temporary file and renamed, so a job
  killed while saving leaves the previous checkpoint intact; a resumed run does
  not log a repeated epoch twice.
- `validation_by_source.csv` holds every validation source at every epoch and
  the log prints the three weakest.
- `--init-weights` starts from an earlier `best.pt`, with or without FiLM. A
  model without FiLM gets a generator that starts as the identity, so training
  starts exactly from that model.
- `--label-tolerance N` leaves an N-pixel ring around every label out of the loss.
- The test overlays are spread evenly over the sources.

Swin encoder option:

- `--encoder swin_t` (or another `swin_*`) builds `HairlineFiLMSwinUNet`: the
  network of the Swin U-Net project with the same FiLM conditioning on its five
  decoder stages. Its parameter names are that project's, so its `best.pt` can
  be passed as `--init-weights`. The learning rate, encoder factor and weight
  decay then default to that project's (2e-4, 0.3, 0.01) instead of the U-Net's
  (3e-4, 0.1, 1e-4).

Checkpoints store the architecture name and arguments, the GSD normalisation,
the GSD table and the augmentation configuration, and
`filmunet.models.load_model_from_checkpoint` rebuilds either network from them.

The dataset stays outside the repository and is supplied by absolute path.
Each fresh job writes into its own `runs/` directory. Resume uses the same
directory with `RESUME=1`. Do not run concurrent jobs into one output folder.

## Inference

There is no full-orthomosaic inference for this model yet. The scripts in
`inference/` belong to the earlier `train_unet_hairline.py` network and do not
pass a GSD. A model loaded with `load_model_from_checkpoint` is called as
`model(images, gsd)` with the GSD in mm/px; without the second argument it
treats the GSD as unknown.

# FiLM U-Net hairline crack training on ASL

## Project layout

```text
configs/training.toml       Core training defaults
configs/gsd.csv             GSD (mm per pixel) of each source
src/filmunet/              GSD, augmentation, data, models, losses, evaluation, plots, training
scripts/                  Conda setup, submission, training CLI, epoch monitor
slurm/train_film.sbatch    ASL CUDA training job
slurm/gpu_check.sbatch     Ten-minute GPU and environment check
docs/ARCHITECTURE.md       Module responsibilities
tests/                    Workflow, trainer and model checks
train_unet_film_hairline.py The original single-file trainer
```

The workflow is that of the Swin U-Net project
(https://github.com/poudelavik1/swinunet); see [architecture](ARCHITECTURE.md)
for what was added to the original script.

## Upload to GitHub from Windows

Repository: https://github.com/poudelavik1/filmunet
For future local updates, run from this project directory:

```powershell
git add .gitignore .gitattributes README.md requirements.txt environment.yml pyproject.toml configs src scripts slurm docs tests train_unet_film_hairline.py inference
git commit -m "Update training code"
git push origin main
```

Do not commit the dataset, ZIP, checkpoints, or credentials. A private repository
requires GitHub authentication when cloning (SSH key or credential helper).

## Server setup (once)

Connect to the ASL login node as for the Swin U-Net project.

```bash
cd ~
[ -d filmunet ] || git clone https://github.com/poudelavik1/filmunet.git filmunet
cd filmunet
bash scripts/setup_conda.sh
```

Setup uses the `swinunet` Conda environment of the Swin U-Net project (Python
3.11, PyTorch 2.9.1 / torchvision 0.24.1 CUDA 12.6 wheels) and creates it if it
does not exist. Set `ENV_NAME` to use another environment, and set the same
`ENV_NAME` (or `ENV_DIR`) when submitting. Jobs do not call `conda activate`,
because `/opt/miniconda3` exists only on the login node; they put
`~/.conda/envs/$ENV_NAME/bin` on `PATH` directly.

Setup also downloads the pretrained ResNet-34 encoder into `~/.cache/torch`, so
compute nodes need no internet. Another encoder needs its own weights cached
from the login node, for example:

```bash
ENCODER=swin_t bash scripts/setup_conda.sh
```

## Check the GPU (optional)

```bash
sbatch slurm/gpu_check.sbatch
squeue -u "$USER"
cat logs/gpu_check_JOBID.out logs/gpu_check_JOBID.err
```

The output shows the GPU, driver, PyTorch CUDA build and whether the encoder
weights are cached, and ends with `GPU check passed.`

## The GSD table

`configs/gsd.csv` gives the GSD of each source in mm per pixel:

```text
pattern,gsd_mm_per_px
^EH104_,0.6511
^RAMA8_EAST-,2.5
```

`pattern` is a regular expression searched in the image file name. The first
matching row gives the GSD, and a file without a match (CRACK500, the photos)
has an unknown GSD, for which the model learns a separate embedding. Write
`[0-9]` rather than `\d`: the submission script checks the patterns with
`grep -E` and the trainer applies them as Python regular expressions. The
values in the file were read from the DXF drawings and the EH104 world file;
its comments say where each one comes from.

Submission prints how many training images each pattern matches and refuses to
queue a FiLM run when none match, because without GSD values the conditioning
has nothing to learn from. `GSD_CSV=/absolute/path.csv` uses another table.

## Submit CUDA training

The dataset must contain `train/images`, `train/masks`, `val/images`,
`val/masks`, `test/images`, and `test/masks`.

```bash
bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
```

Core defaults live in `configs/training.toml`. CLI flags take precedence.
Direct training on an allocated compute node:

```bash
python scripts/train.py "$HOME/dataset_20261008/split_80_10_10" --output runs/manual --gsd-csv configs/gsd.csv
```

Default request: ASL-gpu, one GPU, four CPU cores, 32 GB host RAM, two days.
Request a shorter job with `SBATCH_TIMELIMIT=1-00:00:00` before the command.
Submission prints the job ID, result directory, and monitoring commands.
Each new job uses a separate output directory and does not resume old weights.
On ASL a second job of the same user has been removed from the queue while the first was
running, so submit the next run after the last one ends.

Variables in front of the command change the run:

| Variable | Effect | Default |
| --- | --- | --- |
| `EPOCHS`, `BATCH_SIZE`, `WORKERS` | Core settings | `configs/training.toml` (80, 12, 4) |
| `GSD_CSV` | GSD table | `configs/gsd.csv` |
| `NO_FILM=1` | No GSD conditioning: the ablation baseline | FiLM on |
| `FILM_SKIPS=1` | Also condition the five skip connections | decoder blocks only |
| `AUG_PRESET` | `reference`, `none`, `light`, `medium`, `heavy`, `all` | `reference` |
| `LOSS` | `hairline` (BCE + Tversky + clDice) or `focal` | `hairline` |
| `ENCODER` | `resnet18`, `resnet34`, `resnet50`, or `swin_t` and the other Swin encoders | `resnet34` |
| `GSD_SANITY_CHECK=0` | Skip the test passes with a wrong and a missing GSD | check on |
| `SELECT_PATTERN`, `TARGET_WEIGHT` | Select the model and threshold on one source | all sources |
| `INIT_WEIGHTS`, `LEARNING_RATE` | Fine-tune an earlier `best.pt` | ImageNet encoder |
| `LABEL_TOLERANCE` | Pixels around each label left out of the loss | 0 |
| `SAVE_PREDICTIONS` | Test overlays to save | 40 |

For example, the FiLM ablation is two runs, one after the other:

```bash
bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
NO_FILM=1 bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
```

`ENCODER=swin_t` trains the Swin-Unet of the Swin U-Net project with the same
GSD conditioning. Its `best.pt` from that project can be the starting point:

```bash
ENCODER=swin_t INIT_WEIGHTS=/absolute/path/to/swin/run/best.pt LEARNING_RATE=5e-5 \
  bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
```

By default the best checkpoint and the threshold are chosen on the validation
images of every source pooled by pixel, which the thick-crack sources dominate.
`SELECT_PATTERN` (a regex on validation file names) chooses them on a target
source instead and samples that source's training images `TARGET_WEIGHT` times
as often (default 1.5). For the row/column tiles:

```bash
SELECT_PATTERN='_r[0-9]+_c[0-9]+' bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
```

Only use `SLURM_ACCOUNT` / `SLURM_PARTITION` values confirmed by the
administrator. A pending job can show `InvalidAccount` on this server and still
start on its own. Pending jobs produce no epoch metrics until they start.

Reduce batch size if GPU memory is exhausted. Evaluation also uses tiled images;
if evaluation exhausts memory, reduce `--eval-tile` in the job script.
Early stopping defaults to 15 epochs without improvement, so training can stop
before the requested epoch count.

## Monitor performance

Use the actual job ID and output path printed by the submission script:

```bash
squeue -u "$USER"
tail -F logs/film_unet_JOBID.out
python scripts/monitor_epochs.py /absolute/path/to/run --watch
cat logs/film_unet_JOBID.err
```

Ctrl+C stops the viewer; the submitted job keeps running after SSH disconnects.
The monitor prints a row only after training AND validation finish an epoch.
All epoch values remain in `training_history.csv`, and `validation_by_source.csv`
holds the same metrics for every source separately (the log prints the three
weakest after each epoch).

At the start the log prints the source table with the GSD of each source, the
number of training tiles whose GSD is known, and the FiLM configuration. After
training, the run directory holds:

- `best.pt`, resumable `last.pt`, `configuration.json`, `training_history.csv`,
  `validation_by_source.csv`, `val_threshold_sweep.csv`
- `test_metrics.json`: pooled, per source and per GSD value
- `gsd_sanity_check.json` and `plots/gsd_sanity_check.png`: the test metrics
  with the true GSD, half, double and none. A model that uses the GSD loses
  accuracy when it is told a wrong one.
- `augmentation_counts.json`, `contrast_coverage.json`, `test_predictions/`,
  `plots/`, and the Git commit and package versions of the job

## Resume and update

Resume into the same run directory using its checkpoint:

```bash
RESUME=1 bash scripts/submit.sh /absolute/path/to/split_80_10_10 /absolute/path/to/existing/run
```

Pass the same variables as in the first submission. Never submit two jobs
writing into the same output directory. Cancel a job using `scancel JOBID`.
Update the repo only after active jobs finish:

```bash
git pull --ff-only
```

Retrieve results with `scp -r` from the server, as for the Swin U-Net project.

ASL instructions: https://asl.ait.ac.th/package-slurm.html
Slurm examples: https://asl.ait.ac.th/package-script.html
PyTorch wheels: https://pytorch.org/get-started/previous-versions/

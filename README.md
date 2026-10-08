# FiLM U-Net hairline crack segmentation

Training for thin and faint concrete cracks with a U-Net that is told the ground
sampling distance (GSD, mm per pixel) of every tile through FiLM conditioning:
native-resolution crops, ResNet encoder (Swin optional), BCE + Tversky + clDice,
tolerant crack metrics, and an extended augmentation library that keeps the GSD
right through every scale change.

This is the server version of `train_unet_film_hairline.py`, organised like the
[Swin U-Net project](https://github.com/poudelavik1/swinunet) and run the same way.

```text
configs/             Training defaults and the GSD table
src/filmunet/        GSD, augmentation, data, model, loss, evaluation, plotting, training modules
scripts/             Conda setup, training CLI, submission, epoch monitoring
slurm/               ASL GPU check and training jobs
inference/           Crack detection and AutoCAD drawing of the earlier U-Net (not adapted to this model)
tests/               Workflow, trainer and model checks
docs/                Architecture and server guide
train_unet_film_hairline.py   The original single-file trainer, kept for reference and Kaggle
```

- [ASL setup, submission, monitoring, and resume](docs/ASL_SETUP.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Training defaults](configs/training.toml) and [GSD table](configs/gsd.csv)

## First time on the server

```bash
cd ~
[ -d filmunet ] || git clone https://github.com/poudelavik1/filmunet.git filmunet
cd filmunet
bash scripts/setup_conda.sh
```

Setup reuses the `swinunet` Conda environment of the Swin U-Net project and
caches the pretrained ResNet-34 encoder, which the compute nodes cannot download.

## Train

```bash
cd ~/filmunet
git pull --ff-only
bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
```

Submission checks the dataset, the GSD table and the cached encoder weights
before queueing, then prints commands for following the log and epoch metrics.
Dataset and results are kept out of Git. Cancel or finish old jobs before
updating their code checkout.

The same command with a variable in front changes the run, for example:

```bash
AUG_PRESET=medium bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
NO_FILM=1 bash scripts/submit.sh "$HOME/dataset_20261008/split_80_10_10"
```

`NO_FILM=1` trains the same network without the GSD conditioning, the baseline
of the FiLM ablation. All variables are listed in the [server guide](docs/ASL_SETUP.md).

Run the tests with `python -m unittest discover -s tests -v` and the trainer's
own checks with `python scripts/train.py --self-test`.

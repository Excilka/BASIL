# BASIL

Reference implementation of the semi-supervised retinal layer segmentation method
described in our paper. The package `cmsir/` holds the complete method:

- **Bidirectional A-Scan crossover** (`cmsir/losses.py`, `cmsir/alignment.py`):
  a continuous vertical window is exchanged between a labeled and an unlabeled
  B-scan in both directions, after an image-only integer axial translation.
- **CMSIR** (`cmsir/losses.py`, `cmsir/model.py`): the per-column source
  discriminator trained through a gradient reversal layer, and the layer-wise
  cross-source prototype consistency constraint on encoder features.

The lightweight U-Net in `cmsir/model.py` is intentionally generic. It is shared
verbatim by every compared method so that the comparison isolates the training
objective rather than the backbone.

## Installation

```bash
pip install -r requirements.txt
```

A CUDA device is required: `train.py` and `test.py` call `nvidia-smi` and abort if
no GPU is visible or if free memory is below `runtime.minimum_free_memory_mib`.

## Data layout

Every dataset is read through the same directory convention. `<root>` is the
`data.root` entry of the configuration:

```
<root>/
  train/img/*.bmp     train/mask/*.bmp
  eval/img/*.bmp      eval/mask/*.bmp
  test/img/*.bmp      test/mask/*.bmp
```

`data.image_dir`, `data.mask_dir`, and `data.extension` rename these, and
`data.train_split` / `data.eval_split` / `data.test_split` rename the split
folders. Annotations are single-channel grayscale; `data.mask_values` maps
intensity to a contiguous class index and `data.ignore_value` maps to
`data.ignore_index`. Images and masks are letterboxed to `data.image_size` with
bilinear and nearest-neighbour resampling respectively, padded with
`data.mask_padding_value`.

## Running

Each directory under `examples/` is self-contained. The configuration reads the
frozen split next to it and writes its results into that same directory, so a run
needs no extra arguments:

```bash
cp -r examples/mgu_10pct runs/mgu_10pct
python train.py --config runs/mgu_10pct/config.yaml --test-after-train
```

Set `data.root` (and `data.external_unlabeled_dir` where the configuration uses
one) before the first run.

Training writes `provenance.json`, `split_manifest.json`, `train/metrics.json`,
`train/history.json`, and `train/model/{best,last}.pth`. It refuses to start if
`train/model/last.pth` already exists. Testing an existing checkpoint:

```bash
python test.py --config runs/mgu_10pct/config.yaml \
    --checkpoint runs/mgu_10pct/train/model/best.pth
```

Testing writes `test/metrics.json`, `test/test_log.txt`, and up to
`evaluation.visualization_count` panels in `test/Visualization/`. Dice, mIoU and
mPA come from a dataset-level confusion matrix macro-averaged over foreground
classes; HD95 and mASSD pool symmetric surface distances per class and are
expressed in pixels at `data.image_size`.

## Splits

`examples/*/split.json` lists the exact file names of the labeled, unlabeled,
validation and test sets used for the reported numbers. It is the authoritative
definition of the split; the loader resolves those names against `data.root` and
verifies that the counts match `data.expected_counts`. A copy of what the loader
actually used is written to `split_manifest.json` in the output directory, so a
mismatch is visible immediately.

Both shipped configurations were checked to reproduce their frozen split
file-for-file.

To generate a split from a seed instead, drop `data.split_manifest` and set
`data.labeled_group_count` together with `data.split_seed`. The training seed
itself is `seed` at the top level.

## Reporting protocol

If a `reporting.yaml` sits next to the configuration, `test.py` loads it. It
remaps hard labels for evaluation and visualization only, without touching stored
weights or on-disk masks:

```yaml
label_map: [0, 1, 2, 3, 4, 0]  # class 5 is merged into background when scoring
class_names: [BG, Region1, Region2, Region3, Region4]
training_num_classes: 6  # training still predicts all 6 classes
checkpoint_selection: foreground_1_to_5  # selection rule, unchanged by reporting
```

## Scope of this release

Included: the method, its training and evaluation entry points, and two example
configurations with the exact hyperparameters used for the reported results.

Not included:

- **Trained weights.** Not distributed with this release.
- **Baseline implementations.** The compared methods (Mean Teacher, CPS,
  SD-LayerNet, BCP, DyCON) are third-party work with their own licenses. We do
  not redistribute them; please use the official repositories and cite the
  original papers.
- **Internal orchestration.** The batch scheduler, dataset preparation scripts,
  per-dataset audit tooling, and raw experiment outputs of the research
  repository are omitted. `configs/*.yaml` and `splits/*.json` document the
  resulting protocol.

## Citation

```bibtex
@article{basil,
  title  = {TODO},
  author = {TODO},
  year   = {TODO}
}
```

## License

Released under the MIT License; see `LICENSE`.

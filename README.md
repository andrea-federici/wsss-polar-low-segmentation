# 🌊 Polar Low Segmentation from Weak Pseudo-Labels

Stage 2 of the weakly supervised polar low segmentation project. This repository trains semantic segmentation networks on pseudo-label masks produced by the companion pseudo-label generation pipeline.

The project is built around Hydra configuration, PyTorch Lightning training, and interchangeable segmentation backbones such as SegFormer, UPerNet, DPT, UNetFormer, and `segmentation_models.pytorch` models.

## 🧭 Project Role

The full thesis pipeline has two parts:

```text
image-level labels
    -> wsss-polar-low-pseudolabels
    -> ordinal or binary pseudo-masks
    -> wsss-polar-low-segmentation
    -> dense polar low predictions
```

This repository is the second part. It does not generate pseudo-labels itself; it consumes image/mask pairs and trains a segmentation model to predict polar low regions directly.

## ✨ Highlights

| Capability | Implementation |
| --- | --- |
| Training entry point | `train.py` |
| Prediction/evaluation entry point | `predict.py` |
| Experiment management | Hydra configs under `config/` |
| Training framework | PyTorch Lightning |
| Default polar low model | SegFormer |
| Main polar low loss | Dynamic bootstrapped Dice + cross-entropy |
| Logging | Neptune or TensorBoard |
| Datasets | Polar lows, breast ultrasound, VOC binary person |

## 🗂️ Repository Layout

```text
.
|-- train.py                         # Main training script
|-- predict.py                       # Checkpoint inference, visualization, metrics
|-- seg_launcher.sh                  # Example iteration sweep launcher
|-- conda_env.yml                    # Conda environment definition
|-- config/
|   |-- default.yaml                 # Base training configuration
|   |-- pl_config.yaml               # Polar low experiment
|   |-- pl_gradcam.yaml              # Polar low Grad-CAM mask baseline
|   |-- cancer_config.yaml           # Breast ultrasound experiment
|   |-- voc_binary.yaml              # VOC binary person experiment
|   |-- dataset/                     # Dataset paths, sizes, labels, augmentation
|   |-- model/                       # SegFormer, UPerNet, DPT, UNetFormer, SMP
|   |-- loss/                        # Dice/CE, dynamic bootstrap, MSE
|   |-- optimizer/                   # Adam, AdamW, SGD
|   |-- lr_scheduler/                # Cosine, ReduceLROnPlateau, multistep, exponential
|   `-- logger/                      # Neptune and TensorBoard backends
|-- src/
|   |-- data/                        # Paired image/mask dataset and Lightning datamodule
|   |-- lightning_modules/           # Segmentation Lightning module
|   |-- models/                      # Model factories and backbones
|   `-- utils/                       # Losses, I/O, CRF, logging, seeds
`-- test/
    `-- test_loss.py                 # Dynamic bootstrap loss smoke test
```

## 🧊 Data Layout

The segmentation dataloader matches images and masks by filename stem. For example, `sample_001.png` in the image directory must have a corresponding `sample_001.png`, `.jpg`, or `.jpeg` in the mask directory.

The polar low config currently points to:

```text
data/pl/
|-- images/
|   |-- train/
|   `-- test/
|       `-- pos/
`-- masks_it6/
```

Configured dataset paths:

| Dataset | Config | Image directory | Mask directory |
| --- | --- | --- | --- |
| Polar lows | `config/dataset/polar_lows.yaml` | `data/pl/images/train` | `data/pl/masks_it6` |
| Polar lows, Grad-CAM baseline | `config/pl_gradcam.yaml` | `data/pl/images/train` | `data/pl/masks_gradcam` |
| Breast ultrasound | `config/dataset/cancer.yaml` | `data/bus/images` | `data/bus/masks_core_seed42_it4` |
| VOC binary person | `config/dataset/voc_binary.yaml` | `data/voc_binary/images` | `data/voc_binary/masks_core_seed42_it6/non_vis` |

For multiclass ordinal pseudo-labels, `dataset.num_labels` should equal the number of valid label values, including background. The base config notes that if pseudo-label generation stopped at iteration `n`, this is typically `n + 2`: background `0`, plus foreground confidence classes `1..n+1`.

## ⚙️ Installation

Create the Conda environment:

```bash
conda env create -f conda_env.yml
conda activate pl_seg
```

The environment includes PyTorch, Lightning, Albumentations, Hugging Face Transformers, MONAI, Rasterio, Neptune, and Hydra plugins.

For Neptune logging:

```bash
export NEPTUNE_API_TOKEN="..."
```

To use TensorBoard instead:

```bash
python train.py --config-name pl_config logger=tb
```

## 🚀 Training

Train the main polar low SegFormer experiment:

```bash
python train.py --config-name pl_config
```

Train with a binary Grad-CAM mask baseline:

```bash
python train.py --config-name pl_gradcam
```

Train on VOC binary person masks:

```bash
python train.py --config-name voc_binary
```

Train on breast ultrasound pseudo-labels:

```bash
python train.py --config-name cancer_config
```

Run a Hydra sweep:

```bash
python train.py --config-name pl_config \
  lr_scheduler=cosine,redplat \
  optimizer.hparams.lr=1e-3,5e-4,1e-4 \
  -m
```

Override a pseudo-label iteration:

```bash
python train.py --config-name pl_config \
  dataset.mask_dir=data/pl/masks_it7 \
  dataset.num_labels=9
```

The helper script `seg_launcher.sh` shows the same idea for VOC masks across several pseudo-label iterations.

## 🔎 Prediction and Evaluation

Run inference from a checkpoint:

```bash
python predict.py --config-name pl_config \
  checkpoint.path=relative/path/to/checkpoint.ckpt
```

`checkpoint.path` is resolved relative to `checkpoint.base_folder`, which defaults to `checkpoints/`.

With a ground-truth folder, `predict.py` computes detailed pixel metrics:

```bash
python predict.py --config-name voc_binary \
  checkpoint.path=your_checkpoint.ckpt \
  predict.gt_folder=data/voc_binary/gt_masks
```

Prediction outputs are written to `out/`:

```text
out/
|-- *_comparison.png          # Side-by-side original, prediction, optional GT
|-- additional/               # Individual rendered panels
`-- masks/                    # Raw predicted masks
```

## 🧠 Models

Model selection is controlled by the `model` Hydra group.

| Config | Backend | Notes |
| --- | --- | --- |
| `model=segformer` | Hugging Face `SegformerForSemanticSegmentation` | Default polar low model, pretrained ADE checkpoint |
| `model=upernet` | Hugging Face UPerNet | Supports configured backbones such as ConvNeXt |
| `model=dpt` | Hugging Face DPT | Dense prediction transformer |
| `model=unetformer` | Local FT-UNetFormer implementation | Remote-sensing-oriented architecture |
| `model=smp` | `segmentation_models.pytorch` | Configurable architecture and encoder |
| `model=unet` | Local diffusion-style UNet | Present in codebase, not the main segmentation baseline |

See `src/models/models.md` for the original model notes and references.

## 📉 Losses

Losses are selected through `config/loss/`.

| Loss | Config | Intended use |
| --- | --- | --- |
| Dice + CE | `loss=dice_ce` or `loss=custom_dice_ce` | Standard semantic segmentation |
| Dynamic bootstrap | `loss=dynamic_bootstrap` | Ordinal pseudo-labels with class-dependent trust |
| MSE | `loss=mse` | Regression-style experiments |

The dynamic bootstrap loss mixes pseudo-label targets with the model's own predictions over training progress. Lower-confidence pseudo-label classes can be trusted less, while high-confidence classes and background can be trusted more.

Key controls:

| Parameter | Meaning |
| --- | --- |
| `loss.hparams.class_weight` | Per-class trust values |
| `loss.hparams.bootstrap_start` | Training progress where bootstrapping starts |
| `loss.hparams.bootstrap_end` | Training progress where full bootstrapping is reached |
| `loss.hparams.bootstrap_final_factor` | Maximum target/prediction mixing strength |

## 📦 Logging and Outputs

Hydra writes each run into:

```text
logs/<dataset>/<model>/<date>/<time>/
```

When `checkpoints=true`, the best checkpoint is saved under the run directory:

```text
logs/.../checkpoints/
```

The monitored metric defaults to `val_f1` with `mode=max`.

## 🛠️ Configuration Notes

- `config/default.yaml` defaults to `dataset: voc`, but there is no `config/dataset/voc.yaml` in the repository. Use explicit experiment configs such as `pl_config`, `voc_binary`, or `cancer_config`.
- The current `config/dataset/polar_lows.yaml` in this working tree sets `num_labels: 1`, while `train.py` asserts `dataset.num_labels >= 2`. For binary masks, set `dataset.num_labels=2`; for ordinal masks from iteration `n`, set `dataset.num_labels=n+2`.
- `train.py` and `predict.py` import `src.data.augmentation`, but that file is not present in the current repository tree. Restore it or add an equivalent module before running training.
- The trainer is configured with `accelerator="gpu"`. On CPU-only machines, change the trainer accelerator or run on a CUDA-enabled environment.

## 🔁 Typical Polar Low Workflow

1. Generate pseudo-labels in `../wsss-polar-low-pseudolabels`.
2. Copy or symlink the final `non_vis` mask directory into `data/pl/`.
3. Set `dataset.mask_dir` and `dataset.num_labels` to match the selected pseudo-label iteration.
4. Train with `python train.py --config-name pl_config`.
5. Run `predict.py` with the best checkpoint to export qualitative panels and raw masks.

Example:

```bash
python train.py --config-name pl_config \
  dataset.mask_dir=data/pl/masks_it6 \
  dataset.num_labels=8 \
  model=segformer \
  loss=dynamic_bootstrap
```

## ✅ Testing

Run the available loss test:

```bash
pytest test/test_loss.py
```

This checks that the dynamic bootstrapping loss can be constructed and executed on a small tensor example.

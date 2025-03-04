Model implementations. 

# Unet diffusion

This is not a unet for segmentation but for a diffusion model

# UnetFormer

Architecture designed for remote sensing images... optical ones tho.

- [repo](https://github.com/WangLibo1995/GeoSeg)
- [paper](https://arxiv.org/abs/2109.08937)

The original paper uses the following config:
- optimizer: AdamW
- lr: 6e-4
- weight_decay: 0.01
- scheduler: CosineAnnealingLR with eta_min=1e-6 (or 0) 
- loss: dice + ce
- epochs: 30, 40, 45, 105 in different datasets

# SegFormer

- [implementation](https://huggingface.co/docs/transformers/model_doc/segformer) (HuggingFace)

- Pretrained models here: https://huggingface.co/models?sort=downloads&search=segformer

| **Model variant** | **Depths**    | **Hidden sizes**    | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
|-------------------|---------------|---------------------|-------------------------|----------------|-----------------------|
| MiT-b0            | [2, 2, 2, 2]  | [32, 64, 160, 256]  | 256                     | 3.7            | 70.5                  |
| MiT-b1            | [2, 2, 2, 2]  | [64, 128, 320, 512] | 256                     | 14.0           | 78.7                  |
| MiT-b2            | [3, 4, 6, 3]  | [64, 128, 320, 512] | 768                     | 25.4           | 81.6                  |
| MiT-b3            | [3, 4, 18, 3] | [64, 128, 320, 512] | 768                     | 45.2           | 83.1                  |
| MiT-b4            | [3, 8, 27, 3] | [64, 128, 320, 512] | 768                     | 62.6           | 83.6                  |
| MiT-b5            | [3, 6, 40, 3] | [64, 128, 320, 512] | 768                     | 82.0           | 83.8                  |

For example, pretrained=``"nvidia/segformer-b2-finetuned-ade-512-512"``

# Upernet

- [implementation](https://huggingface.co/docs/transformers/main/model_doc/upernet) (HuggingFace)

UPerNet is a general framework that supports several backbones. The currently implemented backbones are:

- ``'swin'``
- ``'maskswin'``
- ``'convnext'``
- ``'convnext2'``

Available pre-trained models [here](https://huggingface.co/models?search=upernet)

# DPT

Vision Transformers for Dense Prediction

- [paper](https://arxiv.org/abs/2103.13413)
- [implementation](https://huggingface.co/docs/transformers/main/en/model_doc/dpt) (HuggingFace)

# SMP (segmentation_models.pytorch)

List of available architectures and encoders [here](https://github.com/qubvel/segmentation_models.pytorch#architectures). The encoders are pretrained: default weights are from ``'imagenet'`` but in some cases other options are available (see [here](https://github.com/qubvel/segmentation_models.pytorch#encoders-) for details).
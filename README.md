# polar-low-segmentation
Segmentation of Polar Lows from masks generated with XAI techniques

The list of implemented models is available [here](/saravaseg/models/models.md)

## Hydra usage examples

Perform hyperparameters search through multirun:
```
python train.py lr_scheduler=cosine,redplat optimizer.hparams.lr=1e-3,5e-3,1e-4 -m
```

Use a config different from default:
```
python saravaseg/train.py --config-name=fpn_xception_f1067
```

Do **not** log on Neptune:
```
python saravaseg/train.py logger.offline=True
```
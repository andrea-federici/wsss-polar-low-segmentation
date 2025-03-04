# polar-low-segmentation
Segmentation of Polar Lows from masks generated with XAI techniques

The list of implemented models is available [here](/saravaseg/models/README.md)

## Hydra usage examples

Perform hyperparameters search through multirun:
```
python saravaseg/train.py -m
```

Use a config different from default:
```
python saravaseg/train.py --config-name=fpn_xception_f1067
```

Do **not** log on Neptune:
```
python saravaseg/train.py logger.offline=True
```
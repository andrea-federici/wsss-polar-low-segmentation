#!/usr/bin/env bash

for it in {2..7}; do
    CUDA_VISIBLE_DEVICES=0 python3 train.py --config-name voc_binary dataset.mask_dir="data/voc_binary/masks_stdaer_seed42_it${it}"
done
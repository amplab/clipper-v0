#!/usr/bin/env bash

CLIPPER_MODEL_PATH=/data/sda1/tf_models/cifar10_convnet/tf_checkpoint/model.ckpt-19999 \
  TF_CIFAR_BATCH_SIZE=128 \
  python cifar10_tf_modelwrapper.py


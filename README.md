# Observation-Process Diffusion Experiments

This repository contains code for a CS229 project studying diffusion sampling from the stochastic-localization / observation-process perspective.

## Main experiment

We compare:

1. A standard isotropic DDPM-style baseline.
2. A cluster-observation diffusion model using unsupervised pseudo-labels.

The pseudo-labels are obtained by clustering low-resolution CIFAR-10 image features into 10 groups. No ground-truth CIFAR-10 labels are used for training the generative model.

## Files

- `cifar_observation_diffusion.py`: main training, clustering, and sampling script.
- `run_cifar_colab.ipynb`: Colab controller notebook.
- `results/`: lightweight logs and sample grids.

Large checkpoints are not tracked in Git. They should be shared separately through Google Drive or Git LFS.

## Basic Colab workflow

```bash
python cifar_observation_diffusion.py cluster \
  --data_dir /content/data \
  --out ./runs/clusters_lowres8_k10.joblib \
  --feature_mode lowres \
  --lowres 8 \
  --k 10 \
  --pca_dim 64 \
  --batch_size 512 \
  --num_workers 2

python cifar_observation_diffusion.py train \
  --data_dir /content/data \
  --run_dir ./runs/base_colab_20ep \
  --method baseline \
  --epochs 20 \
  --timesteps 500 \
  --base_channels 64 \
  --batch_size 128 \
  --sample_batch_size 64 \
  --num_workers 2 \
  --amp \
  --log_every 100 \
  --sample_every 999 \
  --save_every 5

python cifar_observation_diffusion.py train \
  --data_dir /content/data \
  --run_dir ./runs/cluster_colab_20ep \
  --method cluster \
  --cluster_file ./runs/clusters_lowres8_k10.joblib \
  --epochs 20 \
  --timesteps 500 \
  --base_channels 64 \
  --batch_size 128 \
  --sample_batch_size 64 \
  --num_workers 2 \
  --amp \
  --log_every 100 \
  --sample_every 999 \
  --save_every 5

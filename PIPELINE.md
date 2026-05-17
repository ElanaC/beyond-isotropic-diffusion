# CIFAR-10 DDPM project pipeline

Always run from **repo root** (`CS229/`):

```bash
cd /Users/elanaaa/CS229   # Mac
cd ~/CS229                # GCP VM
```

## Folder layout (what each part is)

```
CS229/
├── CIFAR10_dataset/cifar-10-batches-py/   # DATA — raw pickles + cifar10_pytorch.py
├── ddpm/                                   # LIBRARY — diffusion, U-Nets, loaders (import only)
├── scripts/                                # ENTRY POINTS — run these (train / sample / eval)
│   ├── setup_cuda_env.sh                   # GCP: source before GPU jobs
│   ├── evaluate.py                         # FID + ResNet metrics
│   ├── baseline/
│   │   ├── train.py                        # unconditional DDPM train
│   │   └── sample.py                       # generate PNGs
│   └── conditional/
│       ├── train.py                        # class-conditioned train
│       └── sample.py                       # generate PNGs + labels.npy
├── outputs/                                # ARTIFACTS — checkpoints & samples
│   ├── baseline/
│   └── conditional/
├── experiments/                            # separate research script (not ddpm/ pipeline)
│   └── cifar_observation_diffusion.py
└── requirements.txt
```

**There is only one script per task** under `scripts/`. (Older duplicate files at repo root were removed.)

| You want to… | Run this (only) |
|--------------|-----------------|
| Train baseline | `python scripts/baseline/train.py` |
| Sample baseline | `python scripts/baseline/sample.py` |
| Train conditional | `python scripts/conditional/train.py` |
| Sample conditional | `python scripts/conditional/sample.py` |
| Evaluate samples | `python scripts/evaluate.py` |

`ddpm/` is imported by those scripts — you do not run files inside `ddpm/` directly.

---

## Phase 1 — Baseline

### Train (50k, 200 epochs)

```bash
python scripts/baseline/train.py --device cuda --epochs 200 --batch-size 128 --num-workers 0
```

### Sample 10k

```bash
python scripts/baseline/sample.py --device cuda \
  --checkpoint outputs/baseline/checkpoints/epoch_0199.pt \
  --num-samples 10000 --batch-size 64
```

### Evaluate

```bash
python scripts/evaluate.py --generated-dir outputs/baseline/samples --device cpu
```

### Class embedding plot (PCA)

ResNet-32 features → 2D PCA; points colored by predicted class; ★ = class mean.

```bash
python scripts/plot_class_embedding.py \
  --generated-dir outputs/baseline/samples \
  --name baseline \
  --max-samples 10000 \
  --output outputs/baseline/samples/class_embedding.png
```

After conditional sampling, compare side-by-side:

```bash
python scripts/plot_class_embedding.py \
  --generated-dir outputs/baseline/samples --name baseline \
  --compare-dir outputs/conditional/samples --compare-name conditional \
  --compare-labels-file outputs/conditional/samples/labels.npy \
  --output outputs/class_embedding_compare.png
```

---

## Phase 2 — Class-conditioned

Same pattern under `scripts/conditional/`; sampling writes `labels.npy`.

---

## GCP

```bash
source scripts/setup_cuda_env.sh   # on VM, before train/sample
```

Use **tmux** for long jobs. Stop VM when idle: `gcloud compute instances stop ddpm-train --zone=us-west1-a`

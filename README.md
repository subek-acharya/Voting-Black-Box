# Voting-Black-Box

A unified research codebase for **black-box adversarial attacks** against a
voting/ensemble image classification system for ballot bubble images. This repository
consolidates several previously separate black-box adversarial attack projects into a single
package that targets an ensemble ("voting") of models: ResNet, VGG, CaiT, a multi-output SVM,
spiking ResNet/VGG voters, and a UNet used for preprocessing variants.

The attacks operate in the black-box setting (query access to model outputs, no gradients) and
span several attack families drawn from the original source repositories.

## Installation

```bash
pip install -r requirements.txt
```

Notes:
- Install a build of `torch` / `torchvision` that matches your CUDA (or CPU) setup.
- `mambavision` is only required if you instantiate the MambaVision model variant; it is
  imported lazily, so the rest of the package works without it.

## Repository structure

```
Voting-Black-Box/
├── README.md
├── requirements.txt
├── .gitignore
│
├── VotingBlackBox/
│   ├── ModelFactory.py          # builds / loads the voting ensemble models
│   ├── constants.py             # experiment constants and checkpoint paths
│   ├── utils.py                 # shared data / model helpers
│   │
│   ├── model_architecture/      # model definitions
│   │   ├── ResNet.py
│   │   ├── VGG.py
│   │   ├── cait.py
│   │   ├── MultiOutputSVM.py
│   │   ├── UNet.py
│   │   ├── spiking_resnet_voter.py
│   │   └── spiking_vgg_voter.py
│   │
│   ├── attacks/                 # black-box attack implementations
│   │   ├── adba/                # ADBA
│   │   ├── rays/                # RayS
│   │   ├── square/              # Square attack (Linf + L2)
│   │   ├── surfree/             # SurFree
│   │   ├── samoo/               # SAMOO / MOAA (multi-objective)
│   │   └── transfer/            # adaptive black-box transfer attack
│   │       └── sub_attacks/
│   │           ├── linf_attack/ # APGD-Linf, FGSM
│   │           ├── l1_attack/   # AutoAttack APGD (L1)
│   │           ├── l2_attack/   # APGD (L2) + data management
│   │           └── l0_attack/   # L0 PGD family
│   │
│   └── experiments/             # runnable experiment scripts (run_*.py)
│
├── checkpoint/                  # model weights (git-ignored)
└── data/                        # datasets (git-ignored)
```

## Models and voting ensemble

`ModelFactory.py` builds the individual models that make up the voting ensemble:

- **ResNet20** (`model_architecture/ResNet.py`)
- **VGG16** (`model_architecture/VGG.py`)
- **CaiT** (`model_architecture/cait.py`)
- **Multi-output SVM** (`model_architecture/MultiOutputSVM.py`)
- **Spiking ResNet-20 voter** (`model_architecture/spiking_resnet_voter.py`)
- **Spiking VGG-16 voter** (`model_architecture/spiking_vgg_voter.py`)

A **UNet** (`model_architecture/UNet.py`) is used as a preprocessing front-end for the
`_unet` experiment variants.

## Attack families

- **ADBA** — `attacks/adba/`
- **RayS** — `attacks/rays/`
- **Square** (Linf and L2) — `attacks/square/`
- **SurFree** — `attacks/surfree/`
- **SAMOO / MOAA** (multi-objective) — `attacks/samoo/`
- **Transfer / adaptive black-box** — `attacks/transfer/`, with per-norm sub-attacks under
  `attacks/transfer/sub_attacks/`:
  - `linf_attack/` — APGD-Linf, FGSM
  - `l1_attack/` — AutoAttack APGD (L1)
  - `l2_attack/` — APGD (L2) plus data management utilities
  - `l0_attack/` — L0 PGD family

## Experiments

Experiment entry points live in `VotingBlackBox/experiments/` as `run_*.py` scripts:

- `run_adba.py`, `run_adba_unet.py`
- `run_rays.py`, `run_rays_unet.py`
- `run_square.py`, `run_square_unet.py`
- `run_surfree.py`, `run_surfree_unet.py`
- `run_samoo.py`, `run_samoo_unet.py`

The `_unet` variants run the corresponding attack against the UNet-preprocessing pipeline.

## Checkpoints and data

Model checkpoints live under `checkpoint/` and datasets live under `data/`. Both directories
are **git-ignored**, so the model weights and datasets are not distributed through this
repository — supply them locally before running any experiment.

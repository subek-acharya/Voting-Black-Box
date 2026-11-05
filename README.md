# Black-Box Adversarial Example Attacks on Machine Learning based Election Systems

We provide black-box adversarial attack framework to run 11 multi-norm adversarial attacks
against eight image classifiers.

# Software Installation

We use the following software packages:

<ul>
  <li>python==3.10</li>
  <li>torch</li>
  <li>torchvision</li>
  <li>numpy</li>
  <li>scipy</li>
  <li>einops</li>
  <li>spikingjelly</li>
  <li>mambavision</li>
</ul>

# Models

We provide the following architectures (built by `ModelFactory` from `model_architecture/`):

<ul>
  <li>ResNet20</li>
  <li>VGG16</li>
  <li>CaiT (class-attention image transformer)</li>
  <li>Multi-output SVM classifier</li>
  <li>Spiking ResNet20 voter</li>
  <li>Spiking VGG16 voter</li>
  <li>UNet denoiser (preprocessing front-end for the <code>_unet</code> attack variants)</li>
</ul>

Checkpoints live under `checkpoint/`, which is git-ignored and supplied locally.

# Attacks

| Family | Type | Notes |
| --- | --- | --- |
| ADBA | Decision-based | Active-set decision-boundary attack |
| RayS | Decision-based | Hard-label ray search |
| Square | Score-based | Random search, Linf and L2 variants |
| SurFree | Decision-based | DCT-based decision attack |
| SAMOO / MOAA | Decision-based | Multi-objective evolutionary attack |
| Transfer | Transfer-based | Adaptive black-box; trains a surrogate by querying the oracle |

Transfer is library code under `attacks/transfer/` with no `run_*` script. Its surrogate
sub-attacks are APGD-Linf, FGSM, L1 APGD / AutoAttack, L2 APGD, and L0 PGD / Linf-PGD / Sigma-PGD.

# Experiments

The `run_*.py` scripts in `VotingBlackBox/experiments/` are the entry points, invoked as modules
(`python -m VotingBlackBox.experiments.run_square`). Each builds targets with `ModelFactory`,
selects class-balanced correctly classified clean samples, runs the attack, and reports robust
accuracy.

<ul>
  <li>Non-UNet (attack the classifier directly on the OnlyBubbles validation set): <code>run_adba</code>, <code>run_rays</code>, <code>run_square</code>, <code>run_surfree</code>, <code>run_samoo</code>.</li>
  <li>UNet variants (wrap the classifier with the UNet denoiser): <code>run_adba_unet</code>, <code>run_rays_unet</code>, <code>run_square_unet</code>, <code>run_surfree_unet</code>, <code>run_samoo_unet</code>.</li>
</ul>

# Repository Layout

```
Voting-Black-Box/
├── README.md
├── requirements.txt
└── VotingBlackBox/
    ├── __init__.py
    ├── ModelFactory.py          # model builders + SNN / logits / UNet wrappers
    ├── constants.py             # checkpoint, dataset, and experiment configs
    ├── utils.py                 # data loaders, evaluation, sample selection
    ├── model_architecture/      # ResNet, VGG, cait, MultiOutputSVM, spiking_*_voter, UNet
    ├── attacks/
    │   ├── adba/  rays/  square/  surfree/  samoo/
    │   └── transfer/            # adaptive black-box library + sub_attacks/{linf,l1,l2,l0}
    └── experiments/             # run_*.py and run_*_unet.py entry points
```

# Dataset

We apply the attacks to the Combined and OnlyBubbles grayscale voting datasets. Non-UNet
experiments run on the OnlyBubbles validation set; the `_unet` variants run on the scanned-bubble
datasets. Datasets live under `data/`, which is git-ignored and supplied locally. The datasets are
available at <https://zenodo.org/records/18344123>.

# Contact

For questions or concerns please contact the author at: subekacharya@uri.edu

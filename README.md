# Black-Box Adversarial Example Attacks on Machine Learning based Election Systems

We provide black-box adversarial attack code — query-based (decision and score) and
transfer-based — against machine-learning voting and ballot-bubble image classifiers. All models
and attacks are implemented in PyTorch. Classifiers operate on grayscale 40×50 bubble images with
two classes (filled vs. not filled). Attacks assume black-box access to model outputs; no
gradients. Each attack is run as an experiment module under `VotingBlackBox/experiments/`.

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation section below: <code>pip install -r requirements.txt</code>.</li>
  <li>Place the model checkpoints under <code>checkpoint/</code> (git-ignored, not shipped).</li>
  <li>Place the datasets under <code>data/</code> (git-ignored, not shipped).</li>
  <li>Run an experiment module from the repository root, for example <code>python -m VotingBlackBox.experiments.run_square</code>.</li>
</ol>

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

Versions are indicative; see <code>requirements.txt</code> for the package list. Install a
<code>torch</code> / <code>torchvision</code> build matching your CUDA or CPU setup.
<code>spikingjelly</code> is required for the spiking voters; <code>mambavision</code> is only
needed for the optional MambaVision hook and is imported lazily. Tested on a CUDA-enabled GPU
environment.

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

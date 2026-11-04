# Black-Box Adversarial Example Attacks on Machine Learning based Election Systems

This repository provides **black-box adversarial attack** code — both query-based
(decision/score only) and transfer-based — against machine-learning voting / ballot-bubble
image classifiers. All models and attacks are implemented in PyTorch. The classifiers operate
on grayscale ballot-bubble images (40×50, 2 classes: filled vs. not filled) and the attacks
assume black-box access: queries to the model outputs, no gradients.

Attacks are run through the experiment scripts under `VotingBlackBox/experiments/`
(for example `python -m VotingBlackBox.experiments.run_square`), not by uncommenting lines in a
single file. Each script builds the target models with `ModelFactory`, selects a class-balanced
set of correctly classified clean samples, runs the attack, and reports robust accuracy.

The attack families provided are:

- **ADBA** — active-set decision-boundary attack (`VotingBlackBox/attacks/adba/`).
- **RayS** — hard-label decision-based ray search (`VotingBlackBox/attacks/rays/`).
- **Square** — score-based random-search attack, L∞ and L2 variants (`VotingBlackBox/attacks/square/`).
- **SurFree** — DCT-based decision attack (`VotingBlackBox/attacks/surfree/`).
- **SAMOO / MOAA** — multi-objective evolutionary attack (`VotingBlackBox/attacks/samoo/`).
- **Transfer** — adaptive black-box transfer attack that trains a surrogate by querying the
  oracle and then applies white-box sub-attacks (APGD-L∞, FGSM, APGD/AutoAttack-L1, APGD-L2, and
  L0-PGD / L0-L∞-PGD / L0-Sigma-PGD). Transfer is **library code** under
  `VotingBlackBox/attacks/transfer/`; it has no `run_*` experiment script. The other five
  families each have a runnable `run_*` script and a `run_*_unet` variant.

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation section below: <code>pip install -r requirements.txt</code>.</li>
  <li>Place the model checkpoints under <code>checkpoint/</code> (the directory is git-ignored and not shipped with the repo).</li>
  <li>Place the datasets under <code>data/</code> (also git-ignored and not shipped).</li>
  <li>Run a chosen experiment script from <code>VotingBlackBox/experiments/</code>, for example <code>python -m VotingBlackBox.experiments.run_square</code>.</li>
</ol>

# Software Installation

We use the following software packages:

<ul>
  <li>python (3.10)</li>
  <li>torch</li>
  <li>torchvision</li>
  <li>numpy</li>
  <li>scipy</li>
  <li>einops</li>
  <li>spikingjelly</li>
  <li>mambavision</li>
</ul>

Versions are indicative — see <code>requirements.txt</code> for the exact package list. Install a
build of <code>torch</code> / <code>torchvision</code> that matches your CUDA (or CPU) setup;
<code>mambavision</code> is only needed if you instantiate the optional MambaVision variant and is
imported lazily. Tested on a CUDA-enabled GPU environment.

# Models

`ModelFactory` (in `VotingBlackBox/ModelFactory.py`) builds the following classifiers from the
definitions in `VotingBlackBox/model_architecture/`:

<ul>
  <li>ResNet20</li>
  <li>VGG16</li>
  <li>CaiT (class-attention image transformer)</li>
  <li>A multi-output SVM classifier</li>
  <li>A spiking ResNet20 voter</li>
  <li>A spiking VGG16 voter</li>
  <li>A UNet denoiser, used as the preprocessing front-end for the <code>_unet</code> attack variants</li>
</ul>

Model checkpoints are expected under `checkpoint/`, which is **git-ignored and not shipped** with
this repository. Supply the weights locally; they can be obtained from the dataset/model release.

# Dataset

The attacks target the Combined and OnlyBubbles grayscale voting datasets. Non-UNet experiments
run on the OnlyBubbles validation set; the `_unet` variants run on the scanned-bubble datasets.
Datasets are expected under `data/`, which is **git-ignored and not shipped** with the repository.
The voting datasets are available at <https://zenodo.org/records/18344123>.

# Contact

For questions or concerns please contact the author at: subekacharya@uri.edu

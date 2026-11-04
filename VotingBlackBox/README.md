# VotingBlackBox package

Core Python package for the black-box adversarial attack study on machine-learning voting /
ballot-bubble image classifiers. It bundles the model builders, shared configuration, data/eval
helpers, the attack families, and the runnable experiments.

## Top-level modules

- `__init__.py` — package marker (empty).
- `ModelFactory.py` — defines the `ModelFactory` class plus three `nn.Module` wrappers:
  - `SNNWrapper` — adds a time dimension (`T=4`), averages the spiking network output over time,
    and resets the spiking net each forward, so spiking models expose a standard
    `[N, C, H, W] -> logits` interface.
  - `LogitsOnlyWrapper` — normalizes dict/tuple model outputs down to a plain logits tensor.
  - `UNetModelWrapper` — runs the input through a UNet denoiser, clamps to `[0, 1]`, then through
    the classifier (Input → UNet → Model), so a UNet + classifier can be attacked as one unit.
  - `ModelFactory.get_model(model_name, checkpoint_path)` dispatches on substrings in `model_name`
    and builds: ResNet20 (`resnet`), VGG16 (`vgg`) / VGG11 (`vgg11`), CaiT (`cait`), the
    multi-output SVM (`svm`, needs `[base_path, multi_path]`), the spiking ResNet20 voter
    (`snn_resnet`), the spiking VGG16 voter (`snn_vgg`), and a pure UNet (`unet`). All classifiers
    use grayscale `40×50` input with `num_classes=2`.
  - `ModelFactory.get_unet_model_wrapper(model_name, model_checkpoint, unet_checkpoint)` builds a
    `UNetModelWrapper` for the `_unet` experiment variants.
  - The factory also contains optional/external hooks that are **not shipped as working models of
    this repo**: `_create_carlini` (references an undefined `CarliniNetwork` symbol — broken),
    `_create_ppnet_v2_direct` (loads an external ProtoPNet / Explainable-AI checkpoint via
    `sys.path` manipulation to a sibling directory), and `_create_mambavision` (needs the external
    `mambavision` package, imported lazily). Treat these as experimental hooks, not provided models.
- `constants.py` — central path/config registry. Defines `BASE_DIR` (the `VotingBlackBox/`
  folder), the `CHECKPOINTS` dict (all under `checkpoint/`), the dataset maps (`VAL_DATASETS`,
  `TRAIN_DATASETS`, `DATASETS`, `SCANNED_DATASETS`, `UNET_CHECKPOINT`), the per-model experiment
  dicts (`EXPERIMENTS_RESNET20`, `_VGG16`, `_SVM`, `_CAIT`, `_SNN_VGG16`, `_SNN_RESNET20`,
  `_XAI_RESNET20`, `_XAI_VGG16`), their `EXPERIMENTS_UNET_*` counterparts, and the aggregates
  `EXPERIMENTS_ALL` and `EXPERIMENTS_UNET_ALL`. Each entry carries `ckpt_path`, `dataset_path`,
  `use_unet`, and (for UNet) `unet_ckpt` + `display_name`. Non-UNet experiments use the OnlyBubbles
  validation set; UNet experiments use the scanned-bubble `.pt` datasets.
- `utils.py` — shared data/model helpers:
  - Voter data loaders: `GetVoterValidation`, `GetVoterValidationCombined`, `GetVoterTraining`,
    `GetVoterTrainingCombined`, `GetVoterTrainingBalanced`.
  - `get_scanned_attack_loader(dataset_path, batch_size)` — loads scanned-bubble datasets (handles
    both `xData/yDataBinary` and `data/binary_labels` formats).
  - Tensor/DataLoader conversion: `DataLoaderToTensor`, `TensorToDataLoader`, `TensorToNumpy`,
    `NumpyToTensor`, `GetOutputShape`, and the `MyDataSet` dataset class.
  - Evaluation/selection: `validateD`, `predictD`, `calculateClasswiseAccuracy`, `GetDataBounds`,
    and `GetCorrectlyIdentifiedSamplesBalanced` (picks a class-balanced set of correctly classified
    samples — used by every experiment to build the clean attack set).

## Subpackages

- `model_architecture/` — model definitions (ResNet, VGG, CaiT, UNet, SVM, spiking voters).
  See [model_architecture/README.md](model_architecture/README.md).
- `attacks/` — the black-box attack families (adba, rays, square, surfree, samoo, transfer).
  See [attacks/README.md](attacks/README.md).
- `experiments/` — runnable `run_*.py` experiment scripts.
  See [experiments/README.md](experiments/README.md).

## How it is used

Pick a model/dataset configuration from `constants.py`, build the target with `ModelFactory`,
and run one of the experiment scripts (e.g. `python -m VotingBlackBox.experiments.run_square`).
Checkpoints under `checkpoint/` and datasets under `data/` are git-ignored and must be supplied
locally.

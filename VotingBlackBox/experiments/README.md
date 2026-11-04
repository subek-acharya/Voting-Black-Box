# experiments

Runnable experiment scripts. Each script imports an attack wrapper and an `EXPERIMENTS_*` config
from `constants.py`, builds the target models with `ModelFactory`, selects a class-balanced set of
correctly classified clean samples with `utils.GetCorrectlyIdentifiedSamplesBalanced`, runs the
attack, measures robust accuracy with `utils.validateD`, appends the result to a `*_results.txt`
file, and saves adversarial samples under `adv_samples/`.

Run any script as a module from the repository root, for example:

```bash
python -m VotingBlackBox.experiments.run_square
```

`__init__.py` is a package marker (empty).

## Scripts

| Script | Attack | Target config |
| --- | --- | --- |
| `run_adba.py` | ADBA (`ADBA_AttackWrapper`) | `EXPERIMENTS_ALL` |
| `run_adba_unet.py` | ADBA | `EXPERIMENTS_UNET_RESNET20` (UNet + ResNet20, scanned data) |
| `run_rays.py` | RayS (`RaySAttack`) | `EXPERIMENTS_ALL` |
| `run_rays_unet.py` | RayS | `EXPERIMENTS_UNET_ALL` (scanned data) |
| `run_square.py` | Square-L∞ (`SquareAttackLinf_Wrapper`) | `EXPERIMENTS_ALL`, sweeping eps in {255, 16, 8, 4}/255 |
| `run_square_unet.py` | Square-L∞ | `EXPERIMENTS_UNET_ALL` (scanned data) |
| `run_surfree.py` | SurFree (`SurFree_AttackWrapper`) | `EXPERIMENTS_ALL` |
| `run_surfree_unet.py` | SurFree | `EXPERIMENTS_UNET_ALL` (scanned data) |
| `run_samoo.py` | SAMOO / MOAA (`SMOO_AttackWrapper`) | `EXPERIMENTS_ALL` |
| `run_samoo_unet.py` | SAMOO / MOAA | `EXPERIMENTS_UNET_ALL` (scanned data) |

## UNet variants

The `_unet` scripts wrap each classifier with the UNet denoiser via
`ModelFactory.get_unet_model_wrapper` and attack the combined UNet + classifier on the
scanned-bubble `.pt` datasets, whereas the non-UNet scripts attack the classifier directly on the
OnlyBubbles validation set.

There is no `run_transfer.py`; the transfer attack is library code (see
[../attacks/transfer/README.md](../attacks/transfer/README.md)).

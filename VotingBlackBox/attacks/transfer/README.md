# Transfer (adaptive black-box)

Adaptive black-box transfer attack. The idea: query the target model (the oracle) to label
augmentation data, train a surrogate ("synthetic") model on those labels, then run white-box
sub-attacks against the surrogate and transfer the resulting adversarial examples back to the
oracle.

This family is **library code only** — there is no `run_transfer.py` experiment script, and this
directory has no package `__init__.py`; import the modules directly.

## Files

- `AttackFactory.py` — six attack entry functions, each running a white-box sub-attack against the
  surrogate and returning an adversarial `DataLoader`:
  - `APGD_Linf(device, model, correctLoader, eps)` — DLR-loss APGD-L∞ (500 steps).
  - `APGD_L1(device, model, correctLoader, eps)` — via `AutoAttackL1` with `apgd-dlr`.
  - `APGD_L2(device, model, correctLoader, eps_max)` — via `autopgd_base.APGDAttack` (L2, DLR).
  - `L0_PGD(device, model, correctLoader, sparsity)`.
  - `L0_Linf_PGD(device, model, correctLoader, eps)`.
  - `L0_Sigma_PGD(device, model, correctLoader, sparsity)`.
- `AttackRunner.py` — orchestration: `run_all_attacks`, plus `evaluate_attack`, `save_adv_samples`,
  and `write_attack_results_to_file`.
- `AttackWrappersAdaptiveBlackBox.py` — the adaptive (Papernot-style) pipeline: `AdaptiveAttack`,
  `TrainSyntheticModel`, `TrainingStep`, and `LabelDataUsingOracle` (trains a surrogate by querying
  the oracle, then transfers the sub-attacks).
- `DataLoaderGiant.py` — the `DataLoaderGiant` class, which accumulates oracle-labeled augmentation
  data across iterations.
- `config.py` — centralized attack hyperparameters: `APGD_LINF_PARAMS`, `APGD_L1_PARAMS`,
  `APGD_L2_PARAMS`, `L0_PGD_PARAMS`, `L0_LINF_PARAMS`, `L0_SIGMA_PARAMS`, and `SAVE_CONFIG`.
- `sub_attacks/` — the per-norm white-box surrogate attacks:
  - `__init__.py` — package marker.
  - `linf_attack/` — `APGD_Linf.py` (APGD with CE and DLR losses:
    `AutoAttackPytorchMatGPUWrapper`, `DLR_AutoAttackPytorchMatGPUWrapper`, plus
    projection/gradient/checkpoint helpers) and `FGSM.py` (`FGSMNativePytorch`). Has `__init__.py`.
  - `l1_attack/` — AutoAttack (L1): `autoattack.py` (`AutoAttackL1`), `autopgd_base.py`
    (`APGDAttack`, `L1_projection`), plus `checks.py`, `other_utils.py`, `state.py`. Has
    `__init__.py`.
  - `l2_attack/` — APGD (L2): `autopgd_base.py` (`APGDAttack`, `APGDAttack_targeted`,
    `L1_projection`), `DataManagerPytorch.py` (dataloader/tensor + validation helpers), plus
    `checks.py`, `other_utils.py`. Has `__init__.py`.
  - `l0_attack/` — L0 PGD family: `L0_PGD.py` (`L0_PGD_AttackWrapper`), `L0_Linf_PGD.py`
    (`L0_Linf_PGD_AttackWrapper`), `L0_Sigma_PGD.py` (`L0_Sigma_PGD_AttackWrapper`), `L0_Utils.py`
    (`perturb_L0_box`, `project_L0_box`, `sigma_map`, `perturb_L0_sigma`, `project_L0_sigma`),
    `utils.py` (dataloader/tensor + eval helpers, `dlr_loss`). Has `__init__.py`.

## How it is used

Call `AttackWrappersAdaptiveBlackBox.AdaptiveAttack` / `TrainSyntheticModel` to build a surrogate
from oracle queries, then use `AttackRunner.run_all_attacks` (which calls the `AttackFactory`
functions with the `config.py` hyperparameters) to run the sub-attacks and record results. There is
no experiment script for this family.

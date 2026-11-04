# SAMOO / MOAA

SAMOO / MOAA, a multi-objective evolutionary black-box attack. It evolves a population of candidate
perturbations, trading off misclassification against perturbation size using a multi-objective
optimizer (NSGA-style non-dominated sorting and crowding distance).

## Files

- `__init__.py` — package marker (empty). Import the wrapper directly:
  `from VotingBlackBox.attacks.samoo.AttackWrapperSMOO import SMOO_AttackWrapper`.
- `AttackWrapperSMOO.py` — `SMOO_AttackWrapper(model, device, dataLoader, config)`: runs the attack
  over the dataset and returns an adversarial `DataLoader`.
- `LossFunctions.py` — `UnTargeted` and `Targeted` loss classes, plus tensor helpers
  `pytorch_switch`, `to_pytorch`, `get_device`.
- `MOAA/` — the multi-objective optimizer:
  - `__init__.py` — package marker.
  - `MOAA.py` — `Attack`, `Population`, and a `p_selection` helper.
  - `Solutions.py` — `Solution`, `fast_nondominated_sort`, `calculate_crowding_distance`,
    `crowding_operator`, `tournament_selection`.
  - `operators.py` — genetic operators `mutation`, `crossover`, `generate_offspring`.

## How it is used

Import `SMOO_AttackWrapper` directly (the package `__init__.py` is empty) and call it with a model,
device, clean `DataLoader`, and config. The `run_samoo` / `run_samoo_unet` experiment scripts drive
it over the configured model/dataset sets.

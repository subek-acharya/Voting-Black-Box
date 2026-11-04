# ADBA

ADBA (active-set decision-boundary attack), a black-box attack that searches along the model's
decision boundary.

## Files

- `__init__.py` — package marker; re-exports `ADBA_AttackWrapper` from `adba_attack`.
- `adba.py` — core attack: `ADBA_Attack(model, device, original_image, label, epsilon, budget, ...)`
  and `ATK_ADBA(...)`, with the helper classes `Block`, `V`, `Iter`, and a `progress_bar` helper.
- `adba_attack.py` — `ADBA_AttackWrapper(model, device, dataLoader, config)`: iterates the dataset
  and calls the core attack, returning an adversarial `DataLoader`.
- `datatools.py` — numeric helpers: `func_rho`, `find_midK_of_k1k2`, `next_ADB`.

## How it is used

Import `ADBA_AttackWrapper` (exported from the package) and call it with a model, device, and clean
`DataLoader`. The `run_adba` / `run_adba_unet` experiment scripts drive it over the configured
model/dataset sets.

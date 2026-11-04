# SurFree

SurFree, a decision-based black-box attack that searches in a DCT (frequency) subspace to reduce
the number of queries needed to find adversarial examples.

## Files

- `__init__.py` — package marker; re-exports `SurFree_AttackWrapper` from `surfree_attack`.
- `surfree.py` — the `SurFree` class, plus the `Basis` class and a `distance` helper.
- `surfree_attack.py` — `SurFree_AttackWrapper(model, device, dataLoader, config)`: iterates the
  dataset and runs the SurFree attack, returning an adversarial `DataLoader`.
- `surfree_utils/` — supporting utilities:
  - `__init__.py` — package marker.
  - `attack.py` — `get_init_with_noise(model, X, y, max_queries=1000)`.
  - `dct.py` — 1D/2D/3D DCT and inverse DCT transforms: `dct1`, `idct1`, `dct`, `idct`, `dct_2d`,
    `idct_2d`, `dct_3d`, `idct_3d`.
  - `utils.py` — `atleast_kdim(x, ndim)`.

## How it is used

Import `SurFree_AttackWrapper` (exported from the package) and call it with a model, device, clean
`DataLoader`, and config. The `run_surfree` / `run_surfree_unet` experiment scripts drive it over
the configured model/dataset sets.

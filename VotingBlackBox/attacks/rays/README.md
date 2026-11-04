# RayS

RayS, a hard-label (decision-based) black-box attack that performs a ray search toward the
decision boundary using only the predicted label.

## Files

- `__init__.py` — package marker; re-exports `RaySAttack` from `AttackWrappersRayS`.
- `AttackWrappersRayS.py` — `RaySAttack(device, model, epsMax, queryLimit, cleanLoader)` wrapper
  and the `RayS` class that implements the hard-label ray search.

## How it is used

Import `RaySAttack` (exported from the package) and call it with the device, model, L∞ budget,
query limit, and a clean `DataLoader`; it returns an adversarial `DataLoader`. The `run_rays` /
`run_rays_unet` experiment scripts drive it over the configured model/dataset sets.

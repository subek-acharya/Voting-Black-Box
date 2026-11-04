# Square Attack

Square Attack, a score-based (soft-label) black-box attack that uses random square-shaped
perturbations. Both an L∞ and an L2 variant are provided.

## Files

- `__init__.py` — package marker; re-exports `SquareAttackLinf_Wrapper` and
  `SquareAttackL2_Wrapper`.
- `square_attack_linf.py` — `SquareAttackLinf_Wrapper(model, device, dataLoader, eps, n_iters,
  p_init, n_classes, targeted, loss_type)` and the core `square_attack_linf(...)`.
- `square_attack_l2.py` — `SquareAttackL2_Wrapper(model, device, dataLoader, eps, n_iters, p_init,
  n_classes, targeted)` and the core `square_attack_l2(...)`.
- `square_attack_utils.py` — shared helpers: `p_selection`, `pseudo_gaussian_pert_rectangles`,
  `meta_pseudo_gaussian_pert`, `quantize_numpy`, `dense_to_onehot`, `predict`, `loss`, `softmax`.

## How it is used

Import `SquareAttackLinf_Wrapper` or `SquareAttackL2_Wrapper` (both exported from the package) and
call it with a model, device, and clean `DataLoader`; it returns an adversarial `DataLoader`. The
`run_square` / `run_square_unet` experiment scripts drive the L∞ wrapper over the configured
model/dataset sets, sweeping several epsilon values.

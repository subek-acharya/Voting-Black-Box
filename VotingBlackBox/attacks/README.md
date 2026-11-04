# attacks

Black-box adversarial attack families used against the voting / ballot-bubble classifiers. Each
subpackage is one family. The query-based wrappers follow a common shape — they take
`(model, device, dataLoader, ...)` and return an adversarial `DataLoader` — while the transfer
family is library code with no single wrapper entry point.

`__init__.py` for this package is empty.

## Families

- **adba** — ADBA (active-set decision-boundary attack). Its `__init__.py` re-exports
  `ADBA_AttackWrapper`. See [adba/README.md](adba/README.md).
- **rays** — RayS, a hard-label decision-based ray-search attack. Its `__init__.py` re-exports
  `RaySAttack`. See [rays/README.md](rays/README.md).
- **square** — Square Attack, a score-based random-search attack with L∞ and L2 variants. Its
  `__init__.py` re-exports `SquareAttackLinf_Wrapper` and `SquareAttackL2_Wrapper`. See
  [square/README.md](square/README.md).
- **surfree** — SurFree, a DCT-based decision attack. Its `__init__.py` re-exports
  `SurFree_AttackWrapper`. See [surfree/README.md](surfree/README.md).
- **samoo** — SAMOO / MOAA, a multi-objective evolutionary attack. Its `__init__.py` is empty, so
  import the wrapper directly:
  `from VotingBlackBox.attacks.samoo.AttackWrapperSMOO import SMOO_AttackWrapper`. See
  [samoo/README.md](samoo/README.md).
- **transfer** — adaptive black-box transfer attack. It trains a surrogate by querying the oracle
  and then applies white-box sub-attacks. This family is **library code only** — there is no
  `run_*` experiment script and no package `__init__.py`; import the modules directly. See
  [transfer/README.md](transfer/README.md).

## How it is used

ADBA, RayS, Square, SurFree, and SAMOO each have a runnable `run_*` (and `run_*_unet`) experiment
script under `VotingBlackBox/experiments/`. The transfer family is driven by its own factory /
runner functions rather than an experiment script.

# Reproducibility contract — first migration

This contract is written before migrating implementation code. The reference is the
existing research working tree, including its uncommitted changes. `tests/fixtures/provenance.json`
records the source commit, individual file hashes, dependency versions, and fixture hashes.
Reference capture is explicit and must never run automatically during tests.

## Acceptance targets

| Target | Acceptance criterion |
| --- | --- |
| Rules | Exhaustive small-domain validity/support checks; exact seeded samples, shapes and dtypes against saved legacy outputs. |
| Discrete encoding | Latin scalar/one-hot round trips and invalid-cell decisions match the legacy outputs. |
| Rendered tasks | Block, Orb and Flower fixed-latent images match exactly in the recorded NumPy/OpenCV environment; decoded symbols and readability decisions match, including a one-bit corruption and a blank image. |
| Language | All four D=36/G=2 spaced prompts match byte-for-byte; parsed/strict/truncated/valid fields match on representative completions; sampled answers match. |
| Memorization | Legacy binary sample/group membership counts match; a separate explicit-count API checks validity, conditional/all-output membership and novel-valid yield. |
| Onsets | Legacy consecutive-crossing and retrospective functions match frozen boundary cases. The image-audit protocol reproduces all 24 cached Block/Orb onsets and censoring bounds. |
| Packaging | Tests run with the source repository unavailable from Python imports. Importing the package does not start experiments, access cluster paths or require LLM/model downloads. |
| Figure pipeline | A standalone Block/Orb clock plot is generated from recomputed tables without cluster access. Numeric agreement is required; PDF bytes are not. |

The first migration covers rules, image/language task utilities, membership evaluation,
onset analysis and one figure pipeline. Model forward/loss, checkpoint loading, optimizer
updates, LLM tokenization/loss masking, full training and the remaining paper figures are
**not yet validated or migrated**. Their tests must precede their migration.

## Important distinctions

- Preserve the legacy product parity convention for {-1,+1}; do not equate it blindly
  with even counts of +1 for odd group sizes. Language uses T-count parity.
- Keep seed/RNG behavior and sample ordering. Fixed seeds alone are insufficient if
  algorithms, set ordering, RNG consumption or dependency versions change.
- Legacy `robust_onset` ignores NaNs in its below-threshold check and has historical
  `min_step` behavior. It remains explicitly named `legacy_robust_onset`.
- The image-audit persistence protocol treats nonfinite smoothed values as failures,
  uses a centered width-3 median with repeated endpoints, and uses >= thresholds.
  The legacy smoothing function returns curves shorter than its window unchanged.
- Consecutive crossing uses strict > by default. Do not replace it with >=.
- Preserve conditional versus unconditional memorization denominators. An undefined
  conditional ratio is not an observed zero; legacy zero conventions stay in legacy APIs.
- Missing observations and right-censored onsets are distinct statuses. Budgets are
  reported separately from observed onset times.
- The paper memorization floor 3/1024 is a fixed rate, including 2048-draw evaluations;
  it is not a three-hit requirement when the denominator is a conditional valid count.

## Running the acceptance suite

```bash
python -m pip install -e '.[images,test,torch,plot]'
python -m pytest -q
python paper/figures/image_clocks.py --out outputs/image_clocks
```

For maintainers only, before migration or to audit the reference implementation:

```bash
GENRULE_LEGACY_ROOT=/path/to/DiffusionAttnConsistency python -m pytest -q tests/test_contract.py
```

Do not regenerate fixtures to make a failing refactor pass. An intended scientific or
numerical change needs an explicit protocol/version update and comparison with old results.

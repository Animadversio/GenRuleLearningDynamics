# Validation — 2026-11-10

The reference contract was committed before implementation migration.

| Check | Result |
| --- | --- |
| Legacy implementation against frozen contracts | 57 passed |
| Refactored package against the same contracts | 57 passed |
| Complete suite, including new metrics/onset/packaging tests | 75 passed |
| Non-editable wheel installed in a temporary virtual environment | Built and installed successfully |
| Complete suite using that installed wheel from outside the repository | 75 passed |
| Discrete, language and Flower example | Passed; Flower latent round trip exact |
| Cached Block/Orb clock reconstruction | All 24 onsets/bounds agree |
| Low-dimensional Block fit | exponent 0.6594795284 (rounded: 0.65947953) |
| Reference source hashes after migration | All unchanged |

Wheel validation used a temporary virtual environment with system site packages for
existing NumPy/Torch/OpenCV/pytest dependencies and an offline `--no-deps` installation.
It verifies packaging and independence from the legacy repository; it is not a claim
that dependencies were downloaded and installed in a fully empty environment.

The generated clock PDF/PNG/SVG were inspected. Their numerical table and fit are saved
alongside the plots under `outputs/image_clocks/`. The renderer deliberately covers the
numerical clock panel, not the full typeset/hand-polished paper figure.

Tests and generation used CPU. The initial single-image timing probe was 0.440 seconds,
projecting about 6.6 seconds for the 15 reference images. No model training or large
remote-storage scan was needed; reusable image fixtures and trajectories are cached.

See `reproducibility.md` and `migration.md` for exact scope and preserved legacy behavior.

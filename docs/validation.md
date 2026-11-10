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


## Second migration — model/training foundations

- Legacy DiT/GPT reference suite: 10 passed before implementation migration.
- The same refactored DiT/GPT suite: 10 passed, including forward/loss/sampling,
  three-step Adam/AdamW updates, callbacks and strict small-checkpoint loading.
- End-to-end GPT and diffusion CLI smoke tests: 2 passed, including final evaluation,
  saved artifacts and TensorBoard logs.
- Complete main-environment suite: 87 passed; the built non-editable 0.2.0 wheel also
  passed all 87 tests from outside the repository.
- Legacy actual-tokenizer/TRL suite: 6 passed before SFT helper migration. The migrated
  helpers and the installed 0.2.0 wheel each passed all 6 tests. This includes 36 answer tokens plus EOS, masked
  prompt/padding tokens, malformed-mask rejection and one actual tiny-model TRL step.
- The reference tokenizer snapshot and tokenizer/source file hashes are stored in
  `tests/fixtures/llm/reference.json`; no pretrained model weights were downloaded.
- DiT forward/backward timing probe: 0.514 seconds, projected <15.4 seconds for the
  short numerical capture after imports. Actual tiny Qwen training step: about 1.1 seconds;
  LLM environment initialization/preprocessing adds roughly one minute on this filesystem.
- All captured source hashes still match the unchanged original research checkout.
- The distribution contains the upstream DiT and EDM license texts.

The model fixtures use small synthetic probe weights with nonzero DiT outputs. They do
not establish equality of long GPU training trajectories or validate every historical
large checkpoint. The Qwen harness uses actual tokenization/TRL with a tiny randomly
initialized model; full Qwen3-1.7B training and generation/transfer trajectories remain
outside this validation. Separate reference dependency constraints are in `environments/`.

All three installed console entry points also passed `--help` startup checks.

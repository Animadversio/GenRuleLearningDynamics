# Migration record

## First migration

| Research source | Official package |
| --- | --- |
| `core/parity_lib.py`, `exact_k_lib.py`, `row_k_lib.py`, `latin_square_lib.py` | `genrule.rules.*` |
| Rule dispatch in `core/flower_rule_lib.py` | `genrule.datasets.latents` |
| Flower palette / renderer / decoder | `genrule.datasets.images.colors`, `flower` |
| `core/orb_grid_lib.py` (including Block cell representation) | `genrule.datasets.images.orb`, `block` |
| `core/switch_rule_lib.py` | `genrule.datasets.language.task`, `prompts`, `parsing`; `light_switch` re-exports public functions |
| Binary membership function from evaluation CLI | `genrule.evaluation.memorization` |
| Legacy onset functions | `genrule.analysis.timescales` |
| Image audit logic | Explicit `persistence_onset` and cached trajectory reader |

The initial changes relocate functions, separate responsibilities, remove script/demo
execution and hard-coded import paths, and delay Torch imports in the NumPy rule modules.
The new metrics API supplements rather than silently replaces legacy evaluation.
Legacy rule modules retain encoding helpers for API continuity; further extraction follows
additional tests. The baseline records working-tree file hashes because the research
checkout contained uncommitted experimental changes. No source files were edited there.

## Preserved behavior and limits

- Product parity counts negative signs in {-1,+1}. For odd G, a zero product-parity target
  is not the same as an even number of positive signs. Light-switch parity counts T.
- The Block decoder uses a color margin, not an absolute reconstruction-distance gate;
  the fixed blank-image test is decoded as a valid constant sequence. This behavior is
  exposed in tests, not changed in a structural refactor.
- Latin validation returns separate row and column masks; both are required for sample
  validity. Global-K validation returns validity and inferred K labels.
- Some legacy APIs assume inputs are already in their discrete domain. Use their documented
  input contracts; the package does not silently strengthen these checks.
- The legacy rendered Latin dispatch in `check_latents` inherits a tuple-return mismatch
  from the source implementation. The image acceptance fixtures cover parity only; this
  path is not certified for rendered Latin tasks. Correcting it requires a separate change.
- Legacy uniqueness samplers and the Latin sampling distribution are preserved. No claim
  of uniform sampling over the full Latin support is introduced by this release.
- `legacy_robust_onset` preserves historical NaN/min-step handling. `persistence_onset`
  reproduces the later image-audit criterion and distinguishes missing terminal data.
- `compute_membership_counts` retains its original packed int64 implementation for small-D
  finite binary inputs. `exact_match_mask` is a separate general row-membership interface.
- Fixed image fixtures validate 15 renders/decodes, including corruptions and blanks, in
  the recorded dependency environment; they are not an exhaustive decoder reliability study.

## Next acceptance gates

1. Model migration: fixed DiT/GPT inputs, noise, timesteps, state dicts and forward/loss
   references, followed by short optimizer-step comparisons and checkpoint loading.
2. LLM migration: exact tokenizer/model revisions, chat-template configuration, answer
   token IDs, EOS inclusion and loss-mask fixtures before moving SFT/evaluation code.
3. Log readers: sample TensorBoard/CSV/JSON records with alignment, repeated-checkpoint
   and missing-tag expectations. The current release reads the portable image CSV cache.
4. Remaining figures: inventory manuscript panels, publish plot-ready tables and provenance,
   and extract notebook-only exports. Do not present hand-polished layouts as byte-identical
   script output.
5. Preserve third-party attribution and licenses when importing DiT/EDM/model code; those
   implementations are not included in this first migration.

Long training runs are not required for this first gate. Before any larger computation,
time a small representative workload, report an ETA, and cache reusable numerical outputs.

## Second migration

- Moved DiT without changing architecture/state keys and split EDM loss/preconditioning/sampling
  from the original score training loop. Added the upstream license texts and notices.
- Extracted the actual GPT optimizer-update body and sampler from the research CLI. Its
  portable CLI retains the training/evaluation schedule, adds output/device/seed options,
  and fixes the existing final-evaluation argument mismatch.
- Added a minimal diffusion CLI for generated parity or supplied normalized image tensors,
  using the tested original trainer. It saves TensorBoard logs and numerical artifacts;
  per-task image evaluation schedules are still a separate migration.
- Split Qwen SFT preprocessing and transfer-evaluation helpers from its CLI. The new CLI
  accepts output roots and revision pinning, records dependency/model metadata, and rejects
  unsupported tiny/full-support configurations before live held-out evaluation.
- Tests cover small CPU probe checkpoints rather than all original trained models. The
  original GPT full-vocabulary sampling and diffusion checkpoint numbering are preserved.

The first model gate above is now covered for DiT/EDM and GPT; the Qwen preprocessing gate
covers actual tokenization, EOS/padding masks and a tiny real trainer step. Remaining work
includes full-run Qwen generation/transfer comparisons, other diffusion architectures,
raw log readers and remaining figure exports.

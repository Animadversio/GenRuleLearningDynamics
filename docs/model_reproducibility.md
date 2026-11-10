# Model and training migration contract

This contract precedes the second implementation migration.

- DiT: a small 2-block CPU model, including patch embedding and unpatchification;
  freeze a nonzero probe state so zero-initialized output layers cannot hide errors.
- EDM: fixed forward inputs/sigmas, seeded unreduced loss, sampler trajectory, and
  three actual legacy training iterations. Compare callback steps, gradients, final
  state and saved checkpoint keys/tensors. Cover Adam and AdamW branches.
- GPT: build the same Hugging Face GPT2 model as the research CLI; freeze logits,
  shifted causal loss, seeded sampling, and three AdamW updates on fixed batches.
- Checkpoint compatibility means strict state-dict loading from legacy-created small
  probe checkpoints. It does not yet certify all historical large checkpoints or
  equality of long GPU training trajectories across software/hardware versions.
- Qwen: use the cached Qwen3-1.7B tokenizer with a recorded snapshot/hash, freeze
  token IDs for raw prompts and leading-space answers, and compare actual TRL
  completion-only batches (including EOS and padding) before moving SFT helpers.
  A tiny randomly initialized Qwen model may exercise the real trainer on CPU;
  no full Qwen3-1.7B weights or GPU fine-tuning are required for this gate.
- Keep current training/sampling semantics and explicitly document pre-existing
  defects instead of silently changing historical evaluation protocols.
- Capture environment versions and fixture/source hashes. Test oracle code lives
  under tests/tools, not in the runtime package. Do not regenerate references to
  make a failed migration pass.

Use a single batch/step timing probe before each short computation. Cache the
small reference states, tensors and JSON metadata; report progress during capture.

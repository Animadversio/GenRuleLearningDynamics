# Third-party attribution

- `genrule/models/diffusion/dit.py` is relocated from this project's research
  implementation, adapted from [Meta's DiT](https://github.com/facebookresearch/DiT).
  Copyright (c) Meta Platforms, Inc. and affiliates. It retains the upstream
  CC BY-NC 4.0 license in `LICENSES/DiT-LICENSE.txt`. This migration preserves the
  architecture and state-dict names; the file location/license pointer changed.
- `genrule/models/diffusion/edm.py` contains research adaptations of NVIDIA EDM's
  loss, preconditioning and sampler. Copyright (c) 2022 NVIDIA CORPORATION &
  AFFILIATES. These adaptations remain under CC BY-NC-SA 4.0, as supplied in
  `LICENSES/EDM-LICENSE.txt`. The original research source cites
  [EDM revision 008a4e5](https://github.com/NVlabs/edm/tree/008a4e5316c8e3bfe61a62f874bddba254295afb).
  Functions were separated into an importable module; arithmetic is unchanged.
- GPT2 and Qwen implementations, tokenizers and TRL trainers are used through
  installed Hugging Face packages rather than vendored here. Qwen model/tokenizer
  weights are not bundled; the tests store derived token IDs and record the
  tokenizer snapshot and file hashes.

Upstream license retrieval URLs and SHA-256 digests are recorded in
`LICENSES/sources.json`. These notices describe the named components and do not
relicense unrelated project files under those licenses.

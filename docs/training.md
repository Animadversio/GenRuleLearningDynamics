# Model and training entry points

## DiT / GPT reference environment

```bash
python -m pip install -e '.[models,images,test,plot]' -c environments/models-reference.txt
python -m pytest -q
```

The model contracts use small fixed CPU probe states. Comparisons cover nonzero DiT
forward, EDM preconditioning/loss/sampling, Adam and AdamW updates, GPT logits/causal
loss/sampling/AdamW updates, and strict state-dict checkpoint loading.

Small CPU CLI runs:

```bash
genrule-train-gpt --out_root outputs/gpt --exp_name smoke --device cpu --seed 123 \
  --sample_num 4 --sample_len 4 --group_size 2 --n_positions 8 \
  --n_embd 16 --n_layer 1 --n_head 2 --nsteps 2 --batch_size 2 \
  --eval_sample_size 4 --print_every 1 --use_tensorboard --save_ckpts

genrule-train-diffusion --out outputs/dit-smoke --sample-num 4 --sample-len 4 \
  --group-size 2 --hidden-size 16 --depth 1 --heads 2 --steps 2 \
  --batch-size 2 --device cpu --eval-samples 4 --sampling-steps 3
```

Follow logs with `tensorboard --logdir outputs`. Each entry point also prints live
progress. The diffusion entry saves full loss arrays, callback history, training data,
model configuration, checkpoints, final samples and final parity metrics.

`genrule-train-diffusion --data tensor.npy` accepts existing normalized NCHW training
inputs, including rendered data. It does not infer normalization or the latent rule from
an image tensor. Its generic path trains/samples; image decoding and rule evaluation use
`genrule.datasets` and are not yet wired into that CLI. It is a portable minimal entry
point, not a replacement for every historical per-task evaluation schedule.

The migrated GPT CLI keeps the original schedule and evaluation semantics. One existing
error is fixed explicitly: final evaluation now receives `(step, model, train_X_tsr, ...)`
instead of the mismatched `(step, loss, model, ...)`. The smoke test verifies that the
run reaches final evaluation and saves its final model/results.

## Preserved model behavior

- EDM wrapper checkpoints have `model.` prefixes; final DiT weights are bare state dicts.
  Load into the corresponding wrapper or model with `strict=True`.
- Legacy diffusion checkpoint filenames count updates from zero, while callback epochs
  count from one. The tests preserve this convention.
- GPT sampling includes its original full 3-token vocabulary, including SOS. The old
  parity/memorization evaluation is preserved; this migration does not silently ban SOS.
- GPT configs retain the original library defaults (including BOS/EOS IDs outside the
  small vocabulary); the manual sampler does not use those special-token settings.
- Long GPU trajectories, all historical large checkpoints, and MLP/UNet/ESM variants
  are not certified by the small DiT/GPT tests.

## Qwen preprocessing reference environment

Use a separate environment with `environments/llm-reference.txt`. The Qwen tests use
its actual cached tokenizer and a tiny randomly initialized Qwen model, not 1.7B weights.
Set `GENRULE_QWEN_TOKENIZER` to a local snapshot directory; the reference snapshot is
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Exact tokenizer file hashes are recorded with
the reference token IDs. Tests use local-only loading and do not download a model.


```bash
python -m pip install -e '.[llm,test]' -c environments/llm-reference.txt
GENRULE_QWEN_TOKENIZER=/path/to/local/snapshot python -m pytest -q tests/test_llm_preprocessing.py
```

A paper-sized SFT command (requires suitable GPU resources) is:

```bash
genrule-train-llm --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --out_root outputs/llm --n_people 36 --group_size 2 --parity_target 0 \
  --n_train 4096 --dressing 0 --answer_format spaced \
  --epochs 500 --batch_size 128 --lr 1e-5 --eval_n 1024 \
  --transfer_n 256 --gen_batch 256 --num_evals 60 --num_ckpts 6
```

This command preserves the existing experiment settings and is provided for subsequent
reproduction; the migration validation did not run the full 1.7B-model training. The CLI
prints the evaluation schedule, writes `eval_history.json` and `run_metadata.json`, and
uses the original W&B instrumentation (disable explicitly with `--no_wandb` if desired).
The fixed revision is the locally cached tokenizer/model snapshot used for this migration;
historical runs did not consistently record revisions, so it is not asserted as a recovered
revision for every historical run.

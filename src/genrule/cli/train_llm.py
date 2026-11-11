"""Qwen Light-switch SFT entry point, with portable output paths and revision pinning."""
import argparse
import json
import os
import numpy as np
import torch
from genrule.datasets.language.light_switch import SwitchTask, support_size, chance_rate, render_prompt
from genrule.training.llm_sft import build_dataset, check_masking, SwitchEval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n_people", type=int, default=36)
    ap.add_argument("--rule", default="parity", choices=["parity", "exact_k"])
    ap.add_argument("--group_size", type=int, default=2)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--parity_target", type=int, default=0)
    ap.add_argument("--dressing", type=int, default=0)
    ap.add_argument("--answer_format", default="spaced",
                    choices=["spaced", "compact"])
    ap.add_argument("--n_train", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--eval_n", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="0 = log-spaced schedule with --num_evals instead")
    ap.add_argument("--num_evals", type=int, default=35)
    ap.add_argument("--num_ckpts", type=int, default=0,
                    help="log-spaced checkpoints to keep (6.5 GB each; 0 = none, "
                         "only final/).  Lets a generation-side harness bug be "
                         "re-evaluated offline instead of costing a full re-run.")
    ap.add_argument("--transfer_n", type=int, default=256)
    ap.add_argument("--gen_batch", type=int, default=256)
    ap.add_argument("--wandb_project", default="switch-rule-sft")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp_name", default=None)
    ap.add_argument("--exp_suffix", default="",
                    help="appended to the auto tag, so a corrected re-run lands "
                         "beside the original instead of overwriting it")
    ap.add_argument('--out_root', default='outputs/llm')
    ap.add_argument('--revision', default=None, help='Pin model/tokenizer revision for reproducibility')
    ap.add_argument('--cpu', action='store_true', help='Use CPU without bf16 (for small local test models)')
    args = ap.parse_args()

    task = SwitchTask(n_people=args.n_people, rule=args.rule,
                      group_size=args.group_size, K=args.K,
                      parity_target=args.parity_target, dressing=args.dressing,
                      answer_format=args.answer_format)
    sup = support_size(task)
    tag = args.exp_name or (
        f"{args.rule}{args.group_size if args.rule == 'parity' else args.K}"
        f"_D{args.n_people}_N{args.n_train}_d{args.dressing}_{args.answer_format}"
        f"{args.exp_suffix}")
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"experiment  {tag}")
    print(f"saving to   {out_dir}")
    print(f"support     {sup:,}    chance {chance_rate(task):.3e}")
    print(f"N_train     {args.n_train:,}  -> memorization baseline N/support = "
          f"{args.n_train / sup:.4f}")
    if args.n_train / sup > 0.25:
        print("  WARNING: N is a large fraction of the support, so a model that "
              "has learned\n  the rule hits the training set at that rate by "
              "chance -- little headroom to\n  detect memorization.")

    if args.n_train >= sup or sup < 512:
        raise ValueError('Live evaluation requires support >=512 and held-out valid answers')
    ds, train_answers = build_dataset(task, args.n_train, args.seed)
    print(f"dataset     {len(ds)} pairs, {len(set(ds['completion']))} distinct answers")
    json.dump({"task": tag, "train_answers": train_answers},
              open(os.path.join(out_dir, "train_set.json"), "w"))

    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cfg = SFTConfig(output_dir=out_dir, num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.lr, logging_steps=10, save_strategy="no",
                    bf16=not args.cpu, use_cpu=args.cpu, report_to=[], seed=args.seed,
                    model_init_kwargs={"revision": args.revision} if args.revision else None,
                    completion_only_loss=True)
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds,
                         processing_class=tok)

    from importlib.metadata import version
    metadata = dict(args=vars(args), model_commit=getattr(trainer.model.config, '_commit_hash', None),
                    dependencies={p: version(p) for p in ('torch','transformers','trl','datasets','accelerate')})
    with open(os.path.join(out_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("\nverifying completion-only masking and EOS:")
    check_masking(trainer, tok, args.n_people, render_prompt(task))

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=tag, dir=out_dir,
                   config=dict(vars(args), support=sup, chance=chance_rate(task),
                               mem_baseline=args.n_train / sup))
    ev = SwitchEval(task, tok, train_answers, n=args.eval_n, out_dir=out_dir,
                    batch=args.gen_batch, transfer_n=args.transfer_n,
                    use_wandb=use_wandb)

    total = int(args.epochs * np.ceil(len(ds) / (args.batch_size * args.grad_accum)))
    if args.eval_every:
        sched = set(range(args.eval_every, total + 1, args.eval_every))
    else:   # log-spaced: tau_rule and tau_mem are orders of magnitude apart
        sched = set(np.unique(np.geomspace(1, max(total, 2),
                                           args.num_evals).astype(int)).tolist())
    sched.add(total)
    print(f"eval schedule ({len(sched)} points): "
          f"{sorted(sched)[:8]} ... {sorted(sched)[-3:]}")

    # Checkpoints are insurance against a *generation*-side harness bug, which
    # saved completions cannot repair.  One model is 6.5 GB and the group has
    # 2.7 TB left, so keeping all ~56 eval points across a 5-run ladder (1.8 TB)
    # is out of the question; these few log-spaced points bracket both
    # transitions (rule at ~1-5 epochs, memorisation at ~26-36) for ~100 GB.
    ck_sched = (set(np.unique(np.geomspace(1, max(total, 2), args.num_ckpts)
                              .astype(int)).tolist()) if args.num_ckpts else set())
    if ck_sched:
        print(f"ckpt schedule ({len(ck_sched)} points, 3.8 GB each in bf16, "
              f"{len(ck_sched) * 3.8:.0f} GB total): {sorted(ck_sched)}")

    class _Cb(TrainerCallback):
        def on_train_begin(self, a, state, control, model=None, **kw):
            ev.run(model, 0)

        def on_step_end(self, a, state, control, model=None, **kw):
            if state.global_step in sched:
                ev.run(model, state.global_step)
            if state.global_step in ck_sched:
                p = os.path.join(out_dir, f"ckpt-{state.global_step:06d}")
                # bf16 *copy* -- halves 6.5 GB to 3.4 GB and loses nothing, since
                # re-evaluation loads bf16 anyway.  Copying rather than casting
                # in place leaves the fp32 master weights the optimizer needs.
                sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
                model.save_pretrained(p, state_dict=sd, safe_serialization=True)
                tok.save_pretrained(p)

        def on_train_end(self, a, state, control, model=None, **kw):
            ev.run(model, state.global_step)

    trainer.add_callback(_Cb())
    print()
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "final"))
    print(f"\ndone.  history -> {out_dir}/eval_history.json")

if __name__=='__main__':
    main()

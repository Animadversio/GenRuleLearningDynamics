"""Legacy Qwen SFT preprocessing and live rule/transfer evaluation.

Raw prompt/completion semantics, leading spaces, EOS checks and evaluation
settings are preserved. Heavy TRL/datasets imports stay inside their callers.
"""
import json
import os
from dataclasses import replace
import numpy as np
import torch
from genrule.datasets.language.light_switch import (SwitchTask, chance_rate,
    format_answer, render_prompt, sample_valid, score_batch, support_size, verify)

def build_dataset(task, n_train, seed=0):
    """N distinct valid answers to one fixed prompt, as prompt/completion pairs.

    Plain `prompt`/`completion` strings put this in TRL's *standard* (as opposed
    to *conversational*) dataset format, so TRL concatenates them verbatim and
    applies **no chat template**.  Everything that later feeds a prompt to the
    model must go through `train_prompt_text`, or it is off-distribution -- see
    the note there.
    """
    from datasets import Dataset
    answers = sample_valid(task, n_train, rng=np.random.default_rng(seed),
                           unique=True)
    prompt = render_prompt(task)
    ds = Dataset.from_dict({"prompt": [prompt] * len(answers),
                            "completion": [format_answer(a, task) for a in answers]})
    return ds, answers


def check_masking(trainer, tok, n_people, prompt):
    """Assert loss covers the answer AND the EOS, and none of the prompt, and
    that the prompt the evaluator will use is the one the model trains on."""
    batch = next(iter(trainer.get_train_dataloader()))
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    kept = labels != -100
    kept_ids = ids[kept].tolist()
    print(f"  sequence length      {len(ids)}")
    print(f"  loss-bearing tokens  {int(kept.sum())} "
          f"({100 * float(kept.float().mean()):.1f}%)")
    print(f"  loss is on           {tok.decode(ids[kept])!r}")
    has_eos = tok.eos_token_id in kept_ids
    print(f"  EOS ({tok.eos_token_id}) in loss  {has_eos}")
    if not has_eos:
        raise SystemExit(
            "\nEOS is NOT in the loss-bearing tokens.  Non-termination is this "
            "model's dominant\nzero-shot failure here -- at any dressed prompt it "
            "emits an unbounded 'T F T F ...'\nand hits the token cap having "
            "committed to nothing -- so training without EOS\nwould not fix the "
            "thing SFT is for.  Append eos_token to each completion.")
    n_expect = n_people + 1
    print(f"  -> {int(kept.sum())} loss tokens for {n_people} bits + EOS "
          f"({'as expected' if int(kept.sum()) == n_expect else 'CHECK THIS'})")

    # The eval prompt must be token-identical to the prefix the model trains on.
    # When it silently was not, a checkpoint that had learned the rule perfectly
    # scored 0.000 and was written up as a training collapse.
    prompt_ids = ids[~kept][:int((~kept).sum())].tolist()
    eval_ids = tok(train_prompt_text(prompt))["input_ids"]
    if prompt_ids[:len(eval_ids)] != eval_ids:
        raise SystemExit(
            f"\nEval prompt does not match the training prefix.\n"
            f"  training prefix : {tok.decode(prompt_ids[:24])!r}...\n"
            f"  eval prompt     : {tok.decode(eval_ids[:24])!r}...\n"
            "Every prompt handed to the model must go through `train_prompt_text`.")
    print(f"  eval prompt == training prefix  True ({len(eval_ids)} tokens)")
    return int(kept.sum())


def train_prompt_text(prompt):
    """The prompt exactly as SFT saw it: raw text, no chat template.

    This is the identity function, and it exists to be impossible to forget.

    `build_dataset` hands TRL plain `prompt`/`completion` strings, which is TRL's
    *standard* dataset format -- it concatenates them and never touches the chat
    template.  Evaluating with `apply_chat_template(..., add_generation_prompt=
    True)` therefore asks the model a question it was never fine-tuned on, and
    the damage is silent and wildly uneven: at G=6 the templated prompt still
    scored 0.969 rule_ok, while at G=9 the *same* checkpoint that scores 1.000
    rule_ok on the raw prompt puts 0.999 on `<|im_end|>` and emits nothing.  That
    read as a G=9 training collapse for an entire ladder.

    Switching to a conversational dataset would be the other consistent choice,
    but raw continuation is also the closer analogue of the from-scratch GPT_mini
    baseline this ladder is compared against, so it keeps the chat scaffolding
    out of the comparison.
    """
    return prompt


def logprob_of(model, tok, prompt_text, completions, batch=32):
    """Mean per-token logprob the model assigns to each completion.

    Sampling only sees memorisation once it changes what comes out; the
    likelihood assigned to the training answers moves first and continuously,
    so this detects the onset earlier and with no rare-event counting.
    """
    import torch
    out = []
    for i in range(0, len(completions), batch):
        chunk = completions[i:i + batch]
        texts = [prompt_text + c for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        plen = len(tok(prompt_text)["input_ids"])
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1].float().log_softmax(-1)
        tgt = enc["input_ids"][:, 1:]
        lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = enc["attention_mask"][:, 1:].clone()
        mask[:, :plen - 1] = 0                      # completion tokens only
        out += ((lp * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist()
    return float(np.mean(out))


class SwitchEval:
    """Sample from the live model and score the rule, including transfer."""

    def __init__(self, task, tok, train_answers, n=1024, out_dir=".",
                 transfer=(1, 2, 3), batch=256, transfer_n=256, use_wandb=False):
        self.task, self.tok, self.n = task, tok, n
        self.train_answers = list(train_answers)
        self.transfer_n, self.use_wandb = transfer_n, use_wandb
        # held-out valid answers the model never saw, for the likelihood gap
        seen = set(train_answers)
        pool, rng = [], np.random.default_rng(12345)
        while len(pool) < 256:
            c = sample_valid(task, 512, rng=rng, unique=True)
            pool += [a for a in c if a not in seen]
        self.heldout = pool[:256]
        self.train_probe = self.train_answers[:256]
        self.train_set = set(train_answers)
        self.support = support_size(task)
        self.base = len(self.train_set) / self.support
        self.transfer, self.out_dir, self.batch = transfer, out_dir, batch
        self.history = []

    def _sample(self, model, dressing, n=None):
        import torch
        n = n or self.n
        t = replace(self.task, dressing=dressing)
        text = train_prompt_text(render_prompt(t))
        outs = []
        for i in range(0, n, self.batch):
            b = min(self.batch, n - i)
            enc = self.tok([text] * b, return_tensors="pt",
                           padding=True).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, do_sample=True, temperature=1.0,
                                   top_p=0.95,
                                   max_new_tokens=3 * self.task.n_people,
                                   pad_token_id=self.tok.pad_token_id)
            outs += self.tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        s = score_batch(outs, t)
        valid = [r["answer"] for r in (verify(x, t) for x in outs) if r["rule_ok"]]
        s["mem_ratio"] = sum(a in self.train_set for a in valid) / max(1, len(valid))
        return s, outs

    def run(self, model, step):
        was_training = model.training
        model.eval()
        row = {"step": int(step)}
        for d in (self.task.dressing,) + tuple(self.transfer):
            s, outs = self._sample(model, d,
                                   None if d == self.task.dressing else self.transfer_n)
            row[f"d{d}"] = s
            if d == self.task.dressing:
                # Every completion, not a 16-sample excerpt.  Three metrics on
                # this task have needed fixing after the fact; keeping the raw
                # text makes a *scoring* bug re-scorable for ~25 MB per run.
                row["examples"] = outs
        # likelihood gap: train answers vs never-seen valid answers
        ptext = train_prompt_text(render_prompt(self.task))
        lp_tr = logprob_of(model, self.tok, ptext,
                           [format_answer(a, self.task) for a in self.train_probe])
        lp_ho = logprob_of(model, self.tok, ptext,
                           [format_answer(a, self.task) for a in self.heldout])
        row["logp_train"], row["logp_heldout"] = lp_tr, lp_ho
        row["logp_gap"] = lp_tr - lp_ho
        self.history.append(row)
        tr = row[f"d{self.task.dressing}"]
        print(f"  [eval @ {step:>5}] fmt {tr['format_ok']:.3f} rule {tr['rule_ok']:.3f} "
              f"strict {tr['strict_ok']:.3f} triv {tr['trivial_share']:.2f} "
              f"uniq {tr['n_unique_valid']:>4} mem {tr['mem_ratio']:.3f} "
              f"(base {self.base:.4f}) | transfer "
              + " ".join(f"d{d}={row['d' + str(d)]['rule_ok']:.2f}"
                         for d in self.transfer), flush=True)
        print(f"           logp train {row['logp_train']:.4f}  heldout "
              f"{row['logp_heldout']:.4f}  GAP {row['logp_gap']:+.4f}", flush=True)
        if self.use_wandb:
            import wandb
            flat = {f"d0/{k}": v for k, v in row[f"d{self.task.dressing}"].items()
                    if isinstance(v, (int, float))}
            for d in self.transfer:
                flat[f"transfer/d{d}_rule_ok"] = row[f"d{d}"]["rule_ok"]
                flat[f"transfer/d{d}_format_ok"] = row[f"d{d}"]["format_ok"]
            flat.update(logp_train=row["logp_train"], logp_heldout=row["logp_heldout"],
                        logp_gap=row["logp_gap"], mem_baseline=self.base)
            wandb.log(flat, step=int(step))
        with open(os.path.join(self.out_dir, "eval_history.json"), "w") as f:
            json.dump(self.history, f, indent=1)
        if was_training:
            model.train()

"""Capture real Qwen tokenizer + TRL masks using legacy task/SFT helpers."""
import argparse,hashlib,importlib.metadata,json,sys,tempfile,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
from llm_oracle import load_legacy,make_probe_trainer
ap=argparse.ArgumentParser();ap.add_argument('--legacy-root',required=True,type=Path);ap.add_argument('--tokenizer',required=True,type=Path);args=ap.parse_args()
out=ROOT/'tests/fixtures/llm';out.mkdir(exist_ok=True)
assert not (out/'reference.json').exists(),'Refusing to overwrite reference'
a=load_legacy(args.legacy_root)
with tempfile.TemporaryDirectory() as temp:
 start=time.perf_counter();trainer,tok,task,answers=make_probe_trainer(a,args.tokenizer,temp)
 print(f'Tokenizer + tiny-model/TRL preprocessing probe: {time.perf_counter()-start:.2f}s; inspecting one batch',flush=True)
 batch=next(iter(trainer.get_train_dataloader()))
 a.check_masking(trainer,tok,36,a.task.render_prompt(task))
 prompts=[]
 from dataclasses import replace
 for d in range(4):
  p=a.task.render_prompt(replace(task,dressing=d))
  prompts.append({'text':p,'ids':tok(a.train_prompt_text(p))['input_ids']})
 completions=[a.task.format_answer(ans,task) for ans in answers]
 data={'snapshot':args.tokenizer.name,'eos_id':tok.eos_token_id,'pad_id':tok.pad_token_id,'answers':answers,
       'prompts':prompts,'completions':completions,'completion_ids':[tok(c,add_special_tokens=False)['input_ids'] for c in completions],
       'batch':{k:v.tolist() for k,v in batch.items() if isinstance(v,torch.Tensor)},
       'environment':{p:importlib.metadata.version(p) for p in ['torch','transformers','trl','datasets','accelerate','tokenizers']}}
 for row,labels in zip(batch['input_ids'],batch['labels']):
  kept=row[labels!=-100];assert len(kept)==37 and kept[-1]==tok.eos_token_id
 # One real trainer step provides a smoke check, not a cross-version bitwise optimizer oracle.
 start=time.perf_counter();result=trainer.train();loss=result.training_loss
 print(f'One tiny Qwen trainer forward/backward: {time.perf_counter()-start:.2f}s; loss {loss:.6f}',flush=True)
 data['tiny_training_step_loss']=float(loss)
 data['tokenizer_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.tokenizer.iterdir() if p.name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']}
 data['source_sha256']={p:hashlib.sha256((args.legacy_root/p).read_bytes()).hexdigest() for p in ['scripts/llm_switch_sft.py','core/switch_rule_lib.py']}
 (out/'reference.json').write_text(json.dumps(data,indent=2)+'\n')
 print('Frozen Qwen token IDs and actual TRL masks',flush=True)

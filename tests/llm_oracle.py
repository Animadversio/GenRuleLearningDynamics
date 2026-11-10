"""Legacy SFT preprocessing helpers and a small real-TRL CPU harness."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def load_legacy(root):
    sys.path.insert(0,str(root))
    from core import switch_rule_lib as task
    source=(Path(root)/'scripts/llm_switch_sft.py').read_text()
    names={'build_dataset','check_masking','train_prompt_text'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'np':np,'torch':torch,**vars(task)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'legacy_sft_helpers','exec'),scope)
    return SimpleNamespace(**{n:scope[n] for n in names},task=task)


def make_probe_trainer(api,tokenizer_path,out):
    from transformers import AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
    from trl import SFTConfig,SFTTrainer
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tok.padding_side='left'
    task=api.task.SwitchTask(n_people=36,group_size=2,parity_target=0,dressing=0,answer_format='spaced')
    ds,answers=api.build_dataset(task,4,seed=123)
    torch.manual_seed(123)
    cfg=Qwen3Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=512,bos_token_id=tok.bos_token_id,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,tie_word_embeddings=True)
    model=Qwen3ForCausalLM(cfg)
    args=SFTConfig(output_dir=str(out),max_steps=1,per_device_train_batch_size=2,
        use_cpu=True,bf16=False,fp16=False,report_to=[],seed=123,
        completion_only_loss=True,gradient_checkpointing=False,
        dataloader_pin_memory=False,save_strategy='no',logging_steps=1)
    trainer=SFTTrainer(model=model,args=args,train_dataset=ds,processing_class=tok)
    return trainer,tok,task,answers

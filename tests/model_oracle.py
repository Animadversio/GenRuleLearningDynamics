"""Read-only access to legacy model/training definitions; no research CLI execution."""
import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
from transformers import GPT2Config, GPT2LMHeadModel


def selected(root,path,names,namespace):
    tree=ast.parse((Path(root)/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    scope=dict(namespace)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),scope)
    return SimpleNamespace(**{n:scope[n] for n in names})


def gpt_step_source(root):
    tree=ast.parse((Path(root)/'scripts/GPT_learn_parity_CLI.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in main.body if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch')
    batch_loop=next(n for n in loop.body if isinstance(n,ast.For))
    statements=[];record=False
    for node in batch_loop.body:
        code=ast.unparse(node)
        if code=='optimizer.zero_grad()':record=True
        if record:statements.append(code)
        if code=='optimizer.step()':break
    assert len(statements)==6,statements
    return 'def train_step(model, optimizer, batch_data):\n'+''.join('    '+s+'\n' for s in statements)+'    return loss.detach()\n'


def load(root):
    sys.path.insert(0,str(root))
    DiT=importlib.import_module('core.DiT_model_lib').DiT
    ns={'torch':torch,'nn':nn,'F':F,'Adam':torch.optim.Adam,'AdamW':torch.optim.AdamW,'trange':trange,'np':np}
    edm=selected(root,'core/diffusion_edm_lib.py',{'EDMLoss','EDMDiTPrecondWrapper','edm_sampler','train_score_model_custom_loss'},ns)
    ar=selected(root,'scripts/GPT_learn_parity_CLI.py',{'sample_sequences_parallel'},ns)
    scope={};exec(gpt_step_source(root),scope)
    # Same config constructor as the legacy main(), without running the CLI.
    def build_gpt(config):return GPT2LMHeadModel(GPT2Config(**config,loss_type='ForCausalLMLoss'))
    return SimpleNamespace(DiT=DiT,edm=edm,train_diffusion=edm.train_score_model_custom_loss,
        build_gpt=build_gpt,train_gpt_step=scope['train_step'],sample_gpt=ar.sample_sequences_parallel)

"""Optional actual-Qwen-tokenizer/TRL integration tests; no pretrained weights."""
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from packaging.version import Version

try:
    ready=Version(importlib.metadata.version('trl'))>=Version('1.9.2')
except importlib.metadata.PackageNotFoundError:
    ready=False
if not ready:
    pytest.skip('Run in the documented LLM environment (TRL >=1.9.2)',allow_module_level=True)
import torch
from llm_oracle import make_probe_trainer,load_legacy
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def llm_bundle(tmp_path_factory):
    path=os.environ.get('GENRULE_QWEN_TOKENIZER')
    if not path:pytest.skip('Set GENRULE_QWEN_TOKENIZER to the recorded local tokenizer snapshot')
    if os.environ.get('GENRULE_LEGACY_ROOT'):
        api=load_legacy(os.environ['GENRULE_LEGACY_ROOT'])
    else:
        from genrule.training import llm_sft as helpers
        from genrule.datasets.language import light_switch as task
        api=SimpleNamespace(task=task,build_dataset=helpers.build_dataset,
            train_prompt_text=helpers.train_prompt_text,check_masking=helpers.check_masking)
    trainer,tok,task,answers=make_probe_trainer(api,path,tmp_path_factory.mktemp('llm'))
    reference=json.loads((ROOT/'tests/fixtures/llm/reference.json').read_text())
    return api,trainer,tok,task,answers,reference

def test_qwen_tokens_and_leading_spaces(llm_bundle):
    api,trainer,tok,task,answers,r=llm_bundle
    assert answers==r['answers'] and tok.eos_token_id==r['eos_id']
    from dataclasses import replace
    for i in range(4):
        text=api.task.render_prompt(replace(task,dressing=i))
        assert text==r['prompts'][i]['text']
        assert tok(api.train_prompt_text(text))['input_ids']==r['prompts'][i]['ids']
    for i,ans in enumerate(answers):
        completion=api.task.format_answer(ans,task)
        assert completion.startswith(' ')
        ids=tok(completion,add_special_tokens=False)['input_ids']
        assert ids==r['completion_ids'][i] and len(ids)==36

def test_actual_trl_mask_and_eos(llm_bundle):
    api,trainer,tok,task,answers,r=llm_bundle
    batch=next(iter(trainer.get_train_dataloader()))
    for key,expected in r['batch'].items():assert batch[key].tolist()==expected
    for ids,labels in zip(batch['input_ids'],batch['labels']):
        selected=ids[labels!=-100]
        assert selected.numel()==37 and selected[-1]==tok.eos_token_id
        assert ids[labels==-100].tolist()==r['prompts'][0]['ids']
    api.check_masking(trainer,tok,36,api.task.render_prompt(task))

@pytest.mark.parametrize('fault',['no_eos','wrong_prefix'])
def test_masking_checks_reject_bad_batch(llm_bundle,fault):
    api,trainer,tok,task,answers,r=llm_bundle
    batch={k:torch.tensor(v) for k,v in r['batch'].items()}
    if fault=='no_eos':batch['labels'][batch['input_ids']==tok.eos_token_id]=-100
    else:batch['input_ids'][0,0]=0
    fake=SimpleNamespace(get_train_dataloader=lambda:[batch])
    with pytest.raises(SystemExit):api.check_masking(fake,tok,36,api.task.render_prompt(task))

def test_real_trl_one_cpu_step(llm_bundle):
    _,trainer,_,_,_,_=llm_bundle
    result=trainer.train()
    assert result.global_step==1 and 0<result.training_loss<100

def test_padding_never_contributes_to_loss(llm_bundle):
    _,trainer,_,_,_,_=llm_bundle
    first=trainer.train_dataset[0]
    short={key:value[:-3] for key,value in first.items() if key in ('input_ids','labels')}
    batch=trainer.data_collator([first,short])
    assert batch['attention_mask'][1,-3:].tolist()==[0,0,0]
    assert batch['labels'][1,-3:].tolist()==[-100,-100,-100]

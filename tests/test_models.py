"""Identical model tests run against legacy definitions before package migration."""
import hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def model_reference():
    torch.set_num_threads(2)
    return torch.load(ROOT/'tests/fixtures/models/reference.pt',weights_only=True,map_location='cpu')

@pytest.fixture(scope='module')
def model_api():
    if os.environ.get('GENRULE_LEGACY_ROOT'):
        from model_oracle import load
        return load(os.environ['GENRULE_LEGACY_ROOT'])
    from genrule.models.diffusion.dit import DiT
    from genrule.models.diffusion import edm
    from genrule.models.autoregressive.gpt import build_gpt,sample_sequences_parallel
    from genrule.training.diffusion import train_score_model_custom_loss
    from genrule.training.autoregressive import train_step
    return SimpleNamespace(DiT=DiT,edm=edm,build_gpt=build_gpt,train_diffusion=train_score_model_custom_loss,
        train_gpt_step=train_step,sample_gpt=sample_sequences_parallel)

def close(got,expected):
    torch.testing.assert_close(got,expected,rtol=1e-5,atol=1e-6)

def assert_state(model,expected):
    assert model.state_dict().keys()==expected.keys()
    for key,val in model.state_dict().items():close(val,expected[key])

def dit_model(a,r):
    model=a.DiT(**r['dit_config']);model.load_state_dict(r['dit_state'],strict=True)
    return model

def test_dit_forward_nonzero(model_api,model_reference):
    a,r=model_api,model_reference;m=dit_model(a,r).eval()
    result=m(r['x'],r['t'],r['labels'])
    assert result.abs().max()>1e-3
    close(result,r['dit_forward'])

def test_edm_forward_loss_sampler(model_api,model_reference):
    a,r=model_api,model_reference
    m=a.edm.EDMDiTPrecondWrapper(dit_model(a,r).eval(),sigma_data=1.)
    close(m(r['x'],r['t'].clone()),r['edm_forward'])
    torch.manual_seed(123);close(a.edm.EDMLoss(sigma_data=1.)(m,r['x']),r['edm_loss'])
    result=a.edm.edm_sampler(m,r['x'],num_steps=4,sigma_min=.01,sigma_max=2,return_traj=True)
    for got,want in zip(result,r['sampler']):close(got,want)

@pytest.mark.parametrize('weight_decay',[0.,.01])
def test_diffusion_training_and_checkpoints(model_api,model_reference,tmp_path,weight_decay):
    a,r=model_api,model_reference
    m=a.edm.EDMDiTPrecondWrapper(dit_model(a,r),sigma_data=1.);records=[]
    torch.manual_seed(321)
    m,losses=a.train_diffusion(r['x'],m,a.edm.EDMLoss(sigma_data=1.),lr=.001,nepochs=3,batch_size=2,device='cpu',callback=lambda **kw:records.append({k:v for k,v in kw.items() if k!='model'}),callback_freq=1,weight_decay=weight_decay,save_ckpts=True,save_ckpt_step_list=[1],ckpt_dir=tmp_path)
    expected=r['diffusion_'+str(weight_decay)]
    close(torch.tensor(losses),expected['losses']);assert_state(m,expected['state'])
    assert [v['epoch'] for v in records]==[1,2,3]
    for got,want in zip(records,expected['callbacks']):
        for key in ('loss','grad_norm'):assert got[key]==pytest.approx(want[key],rel=1e-5,abs=1e-6)
    assert sorted(p.name for p in tmp_path.glob('*.pth'))==sorted(expected['checkpoints'])
    for name,state in expected['checkpoints'].items():
        ck=torch.load(tmp_path/name,map_location='cpu',weights_only=True)
        assert ck.keys()==state.keys()
        for key in ck:close(ck[key],state[key])

def test_gpt_forward_and_independent_shifted_loss(model_api,model_reference):
    a,r=model_api,model_reference;m=a.build_gpt(r['gpt_config']);m.load_state_dict(r['gpt_state'],strict=True);m.eval()
    batch=r['gpt_batches'][0];out=m(input_ids=batch,labels=batch)
    close(out.logits,r['gpt_logits']);close(out.loss,r['gpt_loss'])
    close(out.loss,F.cross_entropy(out.logits[:,:-1].reshape(-1,3),batch[:,1:].reshape(-1)))

def test_gpt_sampling(model_api,model_reference):
    a,r=model_api,model_reference;m=a.build_gpt(r['gpt_config']);m.load_state_dict(r['gpt_state'])
    torch.manual_seed(456)
    close(a.sample_gpt(m,num_samples=4,max_length=4,device='cpu',batch_size=2),r['gpt_samples'])

def test_gpt_adamw_steps(model_api,model_reference):
    a,r=model_api,model_reference;m=a.build_gpt(r['gpt_config']);m.load_state_dict(r['gpt_state']);m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);losses=[]
    for i,batch in enumerate(r['gpt_batches']):
        torch.manual_seed(900+i);losses.append(a.train_gpt_step(m,opt,batch))
    close(torch.stack(losses),r['gpt_train_losses']);assert_state(m,r['gpt_final_state'])
    state=opt.state_dict()
    assert state['param_groups']==r['gpt_optimizer_state']['param_groups']
    for key,entry in state['state'].items():
        for name,value in entry.items():close(value,r['gpt_optimizer_state']['state'][key][name])

@pytest.mark.parametrize('kind',['dit','gpt'])
def test_legacy_checkpoint_strict_load(model_api,model_reference,kind):
    a,r=model_api,model_reference
    m=a.DiT(**r['dit_config']) if kind=='dit' else a.build_gpt(r['gpt_config'])
    state=torch.load(ROOT/f'tests/fixtures/models/{kind}_state.pt',weights_only=True,map_location='cpu')
    m.load_state_dict(state,strict=True);assert_state(m,r[kind+'_state'])

def test_model_fixture_integrity():
    root=ROOT/'tests/fixtures/models';manifest=json.loads((root/'provenance.json').read_text())
    for name,digest in manifest['artifacts'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest

"""The same tests are run on the legacy implementation BEFORE migration, then genrule."""
from dataclasses import replace
import itertools
import numpy as np
import pytest
import torch

@pytest.mark.parametrize('key', ['parity_samples','parity_unique','exact_samples','row_samples','row_variable_samples','global_samples','latin_samples'])
def test_seeded_sampling(api,arrays,key):
    np.random.seed(123)
    calls={
        'parity_samples':lambda:api.parity.sample_group_parity_vec(16,12,3,0),
        'parity_unique':lambda:api.parity.sample_ensuring_uniqueness(12,lambda N:api.parity.sample_group_parity_vec(N,12,3)),
        'exact_samples':lambda:api.exact.sample_exact_k_dataset(12,8,3),
        'row_samples':lambda:api.row.sample_row_k_dataset(12,3,1),
        'row_variable_samples':lambda:api.row.sample_row_variable_k_dataset(12,3,[1,2]),
        'global_samples':lambda:api.row.sample_global_k_dataset(12,3,[1,2]),
        'latin_samples':lambda:api.latin.sample_latin_square_dataset(8,3),
    }
    result=calls[key]()
    if key=='global_samples':
        result,labels=result
        np.testing.assert_array_equal(labels,arrays['global_labels'])
    assert result.dtype==arrays[key].dtype
    np.testing.assert_array_equal(result,arrays[key])
    if key!='parity_samples': assert len(np.unique(result.reshape(len(result),-1),axis=0))==len(result)

@pytest.mark.parametrize('g',[1,2,3,6])
@pytest.mark.parametrize('target',[0,1])
def test_exhaustive_product_parity(api,g,target):
    x=np.array(list(itertools.product([-1,1],repeat=6)))
    valid=(api.parity.parity_func(x.reshape(-1,6//g,g),axis=2)==target).all(axis=1)
    # Independent count of negative signs verifies legacy parity convention.
    expected=((x.reshape(-1,6//g,g)<0).sum(axis=2)%2==target).all(axis=1)
    np.testing.assert_array_equal(valid,expected)
    assert valid.sum()==2**(6-6//g)

@pytest.mark.parametrize('k',range(7))
def test_exhaustive_exact_k(api,k):
    x=np.array(list(itertools.product([-1,1],repeat=6)))
    valid=api.exact.exact_k_check(x,k)
    np.testing.assert_array_equal(valid,(x==1).sum(axis=1)==k)
    assert valid.sum()==api.exact.valid_set_size(6,k)

@pytest.mark.parametrize('kind',['row','variable','global'])
def test_exhaustive_row_rules(api,kind):
    x=np.array(list(itertools.product([-1,1],repeat=9)))
    counts=(x.reshape(-1,3,3)==1).sum(axis=2)
    if kind=='row':
        got=api.row.check_row_k_batch(x,3,1); expected=(counts==1).all(axis=1)
        support=api.row.valid_set_size_row_k(3,1)
    elif kind=='variable':
        got=api.row.check_row_variable_k_batch(x,3,[1,2]); expected=np.isin(counts,[1,2]).all(axis=1)
        support=api.row.valid_set_size_row_variable_k(3,[1,2])
    else:
        got,labels=api.row.check_global_k_batch(x,3,[1,2]); expected=np.isin(counts[:,0],[1,2])&(counts==counts[:,:1]).all(axis=1)
        support=api.row.valid_set_size_global_k(3,[1,2])
    np.testing.assert_array_equal(got,expected)
    assert got.sum()==support

def test_latin_support_and_encoding(api,arrays):
    x=np.array(list(itertools.product(range(2),repeat=4)))
    rows, cols = api.latin.check_latin_square_batch(x,2)
    assert (rows & cols).sum()==api.latin.valid_set_size(2)==2
    lat=arrays['latin_samples']
    np.testing.assert_array_equal(api.latin.encode_latin_square(lat,3),arrays['latin_scalar'])
    np.testing.assert_array_equal(api.latin.int_to_onehot(lat,3),arrays['latin_onehot'])
    np.testing.assert_array_equal(api.latin.onehot_to_int(arrays['latin_onehot'],3),lat)
    np.testing.assert_array_equal(api.latin.snap_to_integer(arrays['latin_corrupt_scalar'],3),arrays['latin_snapped'])
    assert all(v.all() for v in api.latin.check_latin_square_batch(lat,3))

def test_quantization(api,arrays):
    got=api.parity.round_to_pos_neg_one(torch.from_numpy(arrays['quant_input']))
    np.testing.assert_array_equal(got.numpy(),arrays['quant_output'])

@pytest.mark.parametrize('name',['flower','orb','block'])
def test_image_render_decode(api,arrays,name):
    mod=api.flower if name=='flower' else api.orb
    cfg=api.flower.PRESETS['aster24'] if name=='flower' else api.orb.PRESETS['orb36_p8']
    if name=='block': cfg=replace(cfg,content='block')
    images=mod.render_batch(arrays[name+'_latents'],cfg,seed=19)
    np.testing.assert_array_equal(images,arrays[name+'_images'][:-1])
    rec,ok,_=mod.decode_batch(arrays[name+'_images'],cfg)
    np.testing.assert_array_equal(rec,arrays[name+'_decoded'])
    np.testing.assert_array_equal(ok,arrays[name+'_readable'])
    valid=api.flower.check_latents(rec,'parity',cfg,group_size=2)&ok
    np.testing.assert_array_equal(valid,arrays[name+'_valid'])
    assert valid[:3].all() and not valid[3]
    # Legacy Block uses nearest-palette decoding, so even a blank image can
    # map to a rule-valid constant sequence. Preserve and expose that behavior.
    assert bool(valid[-1]) == (name == 'block')

@pytest.mark.parametrize('dressing',range(4))
def test_prompts(api,reference,dressing):
    task=api.switch.SwitchTask(n_people=36,group_size=2,parity_target=0,dressing=dressing,answer_format='spaced')
    prompt=api.switch.render_prompt(task)
    assert prompt==reference['prompts'][dressing]
    assert len(prompt.split())==[35,52,99,177][dressing]

@pytest.mark.parametrize('case',range(10))
def test_language_parser(api,reference,case):
    task=api.switch.SwitchTask(n_people=6,group_size=2,parity_target=0,answer_format='spaced')
    row=reference['language_cases'][case]
    assert api.switch.verify(row['text'],task)==row['expected']

def test_language_sampler(api,reference):
    task=api.switch.SwitchTask(n_people=6,group_size=2,parity_target=0,answer_format='spaced')
    assert api.switch.sample_valid(task,6,rng=np.random.default_rng(123))==reference['language_samples']
    assert len(api.switch.enumerate_valid(task))==api.switch.support_size(task)==8

def test_membership(api,arrays,reference):
    result=api.mem.compute_membership_counts(torch.tensor(arrays['mem_train']),torch.tensor(arrays['mem_gen']),2)
    assert result==reference['membership']
    assert result['sample_mem_num']==2 and result['sample_mem_ratio']==.5

@pytest.mark.parametrize('i',range(8))
def test_legacy_onsets(api,reference,i):
    c=reference['curves'][i]; st=c['steps'];v=c['values']
    np.testing.assert_equal(api.smooth.median_smooth(v),c['smooth'])
    np.testing.assert_equal(api.onset.first_sustained_crossing(st,v,.5,n_consec=2),c['consecutive'])
    np.testing.assert_equal(api.smooth.robust_onset(st,v,.5,min_step=0),c['legacy_persistence'])

@pytest.mark.parametrize('n,support,expected',[(2048,2**18,1/64),(2048,4096,.9),(4096,2**30,3/1024)])
def test_memorization_threshold(api,n,support,expected):
    assert api.onset.mem_threshold(n,support)==expected

"""Explicitly capture small legacy reference outputs, before migrating any code."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from legacy_adapter import load

ap = argparse.ArgumentParser()
ap.add_argument('--legacy-root', type=Path, required=True)
args = ap.parse_args()
out = ROOT / 'tests/fixtures'
assert not (out/'reference.npz').exists(), 'Refusing to overwrite frozen fixtures'
a = load(args.legacy_root)
arrays = {}
metadata = {'cases': {}}

def save(key, value):
    arrays[key] = np.asarray(value)

def seeded(key, fn):
    np.random.seed(123)
    save(key, fn())

seeded('parity_samples', lambda:a.parity.sample_group_parity_vec(16, 12, 3, 0))
seeded('parity_unique', lambda:a.parity.sample_ensuring_uniqueness(12, lambda N:a.parity.sample_group_parity_vec(N, 12, 3)))
seeded('exact_samples', lambda:a.exact.sample_exact_k_dataset(12, 8, 3))
seeded('row_samples', lambda:a.row.sample_row_k_dataset(12, 3, 1))
seeded('row_variable_samples', lambda:a.row.sample_row_variable_k_dataset(12, 3, [1,2]))
np.random.seed(123)
g, k = a.row.sample_global_k_dataset(12, 3, [1,2]);save('global_samples',g);save('global_labels',k)
seeded('latin_samples', lambda:a.latin.sample_latin_square_dataset(8, 3))
x = arrays['latin_samples']; save('latin_scalar',a.latin.encode_latin_square(x,3));save('latin_onehot',a.latin.int_to_onehot(x,3))
# A valid sample, a rule violation, and a failed quantization cell.
scalar = arrays['latin_scalar'].copy();scalar[1,0]=scalar[1,1];scalar[2,0]=0.5
save('latin_corrupt_scalar',scalar);save('latin_snapped',a.latin.snap_to_integer(scalar,3))
quant = torch.tensor([[-1.,1.,0.,float('nan'),.991,-.991,.8]],dtype=torch.float64)
save('quant_input',quant.numpy());save('quant_output',a.parity.round_to_pos_neg_one(quant).numpy())

# Time one render/decode first, before doing the complete small image suite.
start=time.perf_counter()
cfg=a.flower.PRESETS['aster24']
lat=a.flower.sample_latents('parity',1,cfg,seed=123,group_size=2)
im=a.flower.render_batch(lat,cfg,seed=19); a.flower.decode_batch(im,cfg)
elapsed=time.perf_counter()-start
print(f'Timing probe: one Flower render/decode {elapsed:.3f}s; projected 15 images ~{15*elapsed:.1f}s',flush=True)
for i,(name,mod,cfg) in enumerate([
    ('flower',a.flower,a.flower.PRESETS['aster24']),
    ('orb',a.orb,a.orb.PRESETS['orb36_p8']),
    ('block',a.orb,replace(a.orb.PRESETS['orb36_p8'],content='block')),
],1):
    # Oracle and refactor must use identical config and actual latent arrays.
    lat=a.flower.sample_latents('parity',3,cfg,seed=123,group_size=2)
    lat=np.concatenate([lat,lat[:1].copy()]);lat[-1,0,0]^=1
    images=mod.render_batch(lat,cfg,seed=19)
    images=np.concatenate([images,np.zeros_like(images[:1])])
    rec,ok,_=mod.decode_batch(images,cfg)
    save(name+'_latents',lat);save(name+'_images',images);save(name+'_decoded',rec);save(name+'_readable',ok)
    save(name+'_valid',a.flower.check_latents(rec,'parity',cfg,group_size=2)&ok)
    print(f'[{i}/3] captured {name}: {len(images)} images',flush=True)

metadata['prompts']=[]
for dressing in range(4):
    task=a.switch.SwitchTask(n_people=36,group_size=2,parity_target=0,dressing=dressing,answer_format='spaced')
    metadata['prompts'].append(a.switch.render_prompt(task))
task=a.switch.SwitchTask(n_people=6,group_size=2,parity_target=0,answer_format='spaced')
metadata['language_samples']=a.switch.sample_valid(task,6,rng=np.random.default_rng(123))
texts=['F F F F F F','T T F F F F','T F F F F F','Answer: T T F F F F',
       '<think>reasoning</think> F F F F F F','<think>unfinished','FF','TTFFFF and FFFFFF','', 'F F F F F F F']
metadata['language_cases']=[{'text':t,'expected':a.switch.verify(t,task)} for t in texts]

train=np.array([[-1,-1,1,1],[1,1,-1,-1]],dtype=np.int64)
gen=np.array([[-1,-1,1,1],[-1,-1,1,1],[1,1,1,1],[-1,1,-1,1]],dtype=np.int64)
save('mem_train',train);save('mem_gen',gen)
metadata['membership']=a.mem.compute_membership_counts(torch.tensor(train),torch.tensor(gen),2)
curves={'persistent':[0,.2,.6,.7,.8,.9], 'dip':[0,.6,.8,.1,.8,.9],
        'two_dips':[0,.8,.1,.1,.8,.9], 'never':[0,.1,.2,.3,.2,.1],
        'equal':[.5]*6,'nan_middle':[0,.6,float('nan'),.8,.9,.9],
        'nan_final':[0,.6,.8,.9,.9,float('nan')], 'short':[.2,.8]}
metadata['curves']=[]
for name,vals in curves.items():
    st=np.arange(len(vals))*100
    metadata['curves'].append(dict(name=name,steps=st.tolist(),values=vals,
        smooth=a.smooth.median_smooth(vals).tolist(),
        consecutive=a.onset.first_sustained_crossing(st,vals,.5,n_consec=2),
        legacy_persistence=a.smooth.robust_onset(st,vals,.5,min_step=0)))
np.savez_compressed(out/'reference.npz',**arrays)
(out/'reference.json').write_text(json.dumps(metadata,indent=2)+'\n')
source_files=['core/'+n+'_lib.py' for n in ['parity','exact_k','row_k','latin_square','flower_rule','orb_grid','switch_rule','onset']]+['scripts/redecode_flower_samples.py','scripts/parity_memorization_eval_cli.py','scripts/audit_image_clock_figure.py']
# Cached complete curves are cheap to reproduce, unlike rereading remote TensorBoard files.
for name in ['onsets.csv','trajectories.csv']:
    source=args.legacy_root/'outputs/paper_audit_20261107/image_clocks'/name
    dest=ROOT/'paper/data'/('image_'+name);dest.write_bytes(source.read_bytes())
    source_files.append(str(source.relative_to(args.legacy_root)))
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance={'source_commit':subprocess.check_output(['git','-C',str(args.legacy_root),'rev-parse','HEAD'],text=True).strip(),
    'reference_kind':'working tree (may differ from commit); hashes are authoritative',
    'source_sha256':{p:hashfile(args.legacy_root/p) for p in source_files},
    'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['numpy','torch','opencv-python','pytest']}},
    'artifact_sha256':{str(p.relative_to(ROOT)):hashfile(p) for p in [out/'reference.npz',out/'reference.json',ROOT/'paper/data/image_onsets.csv',ROOT/'paper/data/image_trajectories.csv']}}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Reference capture complete',flush=True)

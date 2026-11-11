"""Capture legacy CPU model behavior before migrating model/training code."""
import argparse,hashlib,importlib.metadata,json,sys,tempfile,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from model_oracle import load

ap=argparse.ArgumentParser();ap.add_argument('--legacy-root',required=True,type=Path);args=ap.parse_args()
out=ROOT/'tests/fixtures/models';out.mkdir(exist_ok=True)
assert not (out/'reference.pt').exists(),'Refusing to overwrite model reference'
torch.set_num_threads(2);a=load(args.legacy_root)
dit_cfg=dict(input_size=4,patch_size=2,in_channels=1,hidden_size=16,depth=2,num_heads=2,mlp_ratio=2,class_dropout_prob=0.1,num_classes=1,learn_sigma=False)
gpt_cfg=dict(vocab_size=3,n_positions=12,n_embd=16,n_layer=2,n_head=2,resid_pdrop=.1,embd_pdrop=.1,attn_pdrop=.1)
torch.manual_seed(101);dit=a.DiT(**dit_cfg)
with torch.no_grad():
 for p in dit.parameters():
  if p.requires_grad:p.copy_(torch.randn_like(p)*.05)
clone=lambda m:{k:v.detach().clone() for k,v in m.state_dict().items()}
r={'dit_config':dit_cfg,'gpt_config':gpt_cfg,'dit_state':clone(dit)}
x=torch.linspace(-1,1,32).reshape(2,1,4,4);t=torch.tensor([.2,.7]);labels=torch.zeros(2,dtype=torch.long)
r.update(x=x,t=t,labels=labels)
start=time.perf_counter();dit.eval();r['dit_forward']=dit(x,t,labels).detach();dit.train()
edm=a.edm.EDMDiTPrecondWrapper(dit,sigma_data=1.)
torch.manual_seed(42);probe_loss=a.edm.EDMLoss(sigma_data=1.)(edm,x).mean();probe_loss.backward()
elapsed=time.perf_counter()-start
print(f'Timing probe: DiT forward/backward {elapsed:.3f}s; projected tiny reference suite <{max(5,elapsed*30):.1f}s after imports',flush=True)
dit.load_state_dict(r['dit_state']);dit.eval()
r['edm_forward']=edm(x,t.clone()).detach()
torch.manual_seed(123);r['edm_loss']=a.edm.EDMLoss(sigma_data=1.)(edm,x).detach()
r['sampler']=a.edm.edm_sampler(edm,x,num_steps=4,sigma_min=.01,sigma_max=2,return_traj=True)
print('[1/4] DiT/EDM forward, loss and sampler captured',flush=True)
for wd in [0.,.01]:
 dit.load_state_dict(r['dit_state']);records=[]
 with tempfile.TemporaryDirectory() as ck:
  torch.manual_seed(321)
  model,losses=a.train_diffusion(x,edm,a.edm.EDMLoss(sigma_data=1.),lr=.001,nepochs=3,batch_size=2,device='cpu',callback=lambda **kw:records.append({k:v for k,v in kw.items() if k!='model'}),callback_freq=1,weight_decay=wd,save_ckpts=True,save_ckpt_step_list=[1],ckpt_dir=ck)
  checkpoints={p.name:torch.load(p,map_location='cpu',weights_only=True) for p in Path(ck).glob('*.pth')}
 r['diffusion_'+str(wd)]={'state':clone(model),'losses':torch.tensor(losses),'callbacks':records,'checkpoints':checkpoints}
print('[2/4] actual legacy diffusion training, callbacks and checkpoints captured',flush=True)
torch.manual_seed(123);gpt=a.build_gpt(gpt_cfg);r['gpt_state']=clone(gpt)
batches=[torch.tensor([[2,0,1,1,0],[2,1,0,0,1]]),torch.tensor([[2,1,1,1,1],[2,0,0,0,0]]),torch.tensor([[2,1,0,1,0],[2,0,1,0,1]])]
r['gpt_batches']=batches;gpt.eval();output=gpt(input_ids=batches[0],labels=batches[0]);r['gpt_logits']=output.logits.detach();r['gpt_loss']=output.loss.detach()
torch.manual_seed(456);r['gpt_samples']=a.sample_gpt(gpt,num_samples=4,max_length=4,device='cpu',batch_size=2)
gpt.train();opt=torch.optim.AdamW(gpt.parameters(),lr=.001,weight_decay=.01);losses=[]
for i,batch in enumerate(batches):
 torch.manual_seed(900+i);losses.append(a.train_gpt_step(gpt,opt,batch))
r['gpt_train_losses']=torch.stack(losses);r['gpt_final_state']=clone(gpt);r['gpt_optimizer_state']=opt.state_dict()
print('[3/4] GPT logits, shifted loss, sampling and AdamW updates captured',flush=True)
# Keep a bare legacy-created checkpoint for explicit strict-load tests.
torch.save(r['dit_state'],out/'dit_state.pt');torch.save(r['gpt_state'],out/'gpt_state.pt');torch.save(r,out/'reference.pt')
files=['core/DiT_model_lib.py','core/diffusion_edm_lib.py','scripts/GPT_learn_parity_CLI.py','scripts/llm_switch_sft.py']
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta={'device':'cpu','threads':2,'tolerance':{'rtol':1e-5,'atol':1e-6},'note':'Small nonzero synthetic probe states; not the trained paper checkpoints.',
 'source_sha256':{f:hashfile(args.legacy_root/f) for f in files},
 'environment':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','timm','transformers']},
 'artifacts':{p.name:hashfile(p) for p in out.glob('*.pt')}}
(out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print('[4/4] saved model reference and provenance',flush=True)

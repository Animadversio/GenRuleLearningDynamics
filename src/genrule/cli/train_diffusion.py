"""Portable DiT/EDM training entry point using the unchanged legacy trainer.

Generate parity data or provide an NCHW float32 .npy training tensor. Supplied
image tensors must already use the training normalization (typically [-1,1]).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from genrule.models.diffusion.dit import DiT
from genrule.models.diffusion.edm import EDMLoss, EDMDiTPrecondWrapper, edm_sampler
from genrule.training.diffusion import train_score_model_custom_loss
from genrule.rules.parity import sample_group_parity_vec, sample_ensuring_uniqueness, round_to_pos_neg_one
from genrule.evaluation.metrics import exact_match_mask


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path,help='Optional normalized NCHW .npy tensor')
    ap.add_argument('--sample-num',type=int,default=2048)
    ap.add_argument('--sample-len',type=int,default=36)
    ap.add_argument('--group-size',type=int,default=2)
    ap.add_argument('--parity',type=int,choices=[0,1],default=0)
    ap.add_argument('--patch-size',type=int,default=1)
    ap.add_argument('--hidden-size',type=int,default=384)
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--heads',type=int,default=6)
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--weight-decay',type=float,default=0.)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
    ap.add_argument('--log-every',type=int,default=10)
    ap.add_argument('--eval-samples',type=int,default=64)
    ap.add_argument('--sampling-steps',type=int,default=18)
    args=ap.parse_args()
    if args.steps<1 or args.batch_size<1 or args.eval_samples<1 or args.sampling_steps<2:
        ap.error('steps, batch size, eval samples must be positive; sampling steps >=2')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if args.data:
        x=np.load(args.data,allow_pickle=False).astype(np.float32)
        if x.ndim!=4 or x.shape[2]!=x.shape[3] or not np.isfinite(x).all():
            ap.error('--data must contain finite square NCHW images')
    else:
        side=int(np.sqrt(args.sample_len))
        if side*side!=args.sample_len or args.group_size<1 or args.sample_len%args.group_size:
            ap.error('sample length must be square and divisible by group size')
        support=2**(args.sample_len-args.sample_len//args.group_size)
        if not 0<args.sample_num<=support:ap.error('sample count must be within valid support')
        x=sample_ensuring_uniqueness(args.sample_num,lambda N:sample_group_parity_vec(N,args.sample_len,args.group_size,args.parity))
        x=x.reshape(-1,1,side,side).astype(np.float32)
    if x.shape[-1]%args.patch_size:ap.error('patch size must divide image size')
    args.out.mkdir(parents=True,exist_ok=True)
    np.save(args.out/'training_data.npy',x)
    config=dict(input_size=x.shape[-1],patch_size=args.patch_size,in_channels=x.shape[1],
        hidden_size=args.hidden_size,depth=args.depth,num_heads=args.heads,
        num_classes=0,learn_sigma=False)
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.out/'args.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
    model=DiT(**config)
    wrapped=EDMDiTPrecondWrapper(model,sigma_data=1.)
    history=[];ck=args.out/'ckpts';ck.mkdir(exist_ok=True)
    print(f'TensorBoard: tensorboard --logdir {args.out / "tensorboard"}',flush=True)
    with SummaryWriter(str(args.out/'tensorboard')) as writer:
        def callback(epoch,loss,model,grad_norm=None):
            history.append(dict(step=epoch,loss=loss,grad_norm=grad_norm))
            writer.add_scalar('Training/Loss_Step',loss,epoch)
            if grad_norm is not None:writer.add_scalar('Training/Gradient_Norm',grad_norm,epoch)
            writer.flush()
            (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        wrapped,losses=train_score_model_custom_loss(torch.from_numpy(x),wrapped,EDMLoss(sigma_data=1.),
            lr=args.lr,nepochs=args.steps,batch_size=args.batch_size,device=args.device,
            callback=callback,callback_freq=args.log_every,weight_decay=args.weight_decay,
            save_ckpts=True,save_ckpt_step_list=[],ckpt_dir=ck)
        torch.save(model.state_dict(),args.out/'model_final.pth')
        np.save(args.out/'losses.npy',np.asarray(losses))
        wrapped.eval()
        noise=torch.randn((args.eval_samples,*x.shape[1:]),device=args.device)
        generated=edm_sampler(wrapped,noise,num_steps=args.sampling_steps).cpu()
        np.save(args.out/'samples.npy',generated.numpy())
        if args.data is None:
            discrete=round_to_pos_neg_one(generated,eps=.1).flatten(start_dim=1)
            valid=(discrete.reshape(-1,args.sample_len//args.group_size,args.group_size).prod(dim=-1)==(-1)**args.parity).all(dim=1)
            # Evaluate membership only on readable discrete outputs; report its all-output rate explicitly.
            readable=torch.isfinite(discrete).all(dim=1)
            count=int(exact_match_mask(x.reshape(len(x),-1),discrete[readable].numpy()).sum()) if readable.any() else 0
            metrics={'n_total':args.eval_samples,'n_readable':int(readable.sum()),'n_valid':int(valid.sum()),
                     'sample_validity':float(valid.float().mean()),'training_match_all':count/args.eval_samples}
            writer.add_scalar('Eval/Sample_Accuracy',metrics['sample_validity'],args.steps)
            writer.add_scalar('Eval/Sample_Mem_Ratio',metrics['training_match_all'],args.steps)
            (args.out/'final_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'Saved training artifacts to {args.out}',flush=True)

if __name__=='__main__':main()

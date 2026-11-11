"""Small end-to-end CLI tests protect final evaluation and saved artifacts."""
import json
import sys
from pathlib import Path
import numpy as np
import torch


def test_gpt_cli_finishes_final_evaluation(tmp_path,monkeypatch):
    from genrule.cli.train_gpt import main
    torch.set_num_threads(2)
    monkeypatch.setattr(sys,'argv',['train-gpt','--out_root',str(tmp_path),'--exp_name','tiny',
        '--device','cpu','--seed','123','--sample_num','4','--sample_len','4','--group_size','2',
        '--n_positions','8','--n_embd','16','--n_layer','1','--n_head','2',
        '--nsteps','2','--batch_size','2','--eval_sample_size','4','--print_every','1',
        '--save_ckpts','--num_ckpts','2','--use_tensorboard'])
    main()
    root=tmp_path/'tiny'
    assert (root/'model_final.pth').exists()
    rows=json.loads((root/'eval_results.json').read_text())
    assert rows and rows[-1]['step']==2
    assert len(np.load(root/'training_data.npy'))==4
    assert list((root/'tensorboard').glob('events.out.tfevents.*'))


def test_diffusion_cli_tiny_run(tmp_path,monkeypatch):
    from genrule.cli.train_diffusion import main
    torch.set_num_threads(2)
    monkeypatch.setattr(sys,'argv',['train-diffusion','--out',str(tmp_path),
        '--sample-num','4','--sample-len','4','--group-size','2','--steps','2',
        '--hidden-size','16','--depth','1','--heads','2','--batch-size','2',
        '--device','cpu','--eval-samples','4','--sampling-steps','3'])
    main()
    assert (tmp_path/'model_final.pth').exists()
    history=json.loads((tmp_path/'history.json').read_text())
    assert len(history)==2 and history[-1]['step']==2
    assert np.isfinite([r['loss'] for r in history]).all()
    assert list((tmp_path/'tensorboard').glob('events.out.tfevents.*'))
    assert np.load(tmp_path/'samples.npy').shape==(4,1,2,2)
